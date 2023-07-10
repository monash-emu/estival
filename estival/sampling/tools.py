from typing import Tuple, Dict
from multiprocessing import cpu_count, Pool

import cloudpickle
import pandas as pd
from arviz import InferenceData
import numpy as np

from estival.model import BayesianCompartmentalModel
from estival.utils.parallel import map_parallel

SampleIndex = Tuple[int, int]
ParamDict = Dict[str, float]


def likelihood_extras_for_idata(
    idata: InferenceData, bcm: BayesianCompartmentalModel, n_workers: int = None
) -> pd.DataFrame:
    """Calculate the likelihood extras (ll,lprior,lpost + per-target) for all
    samples in supplied InferenceData, returning a DataFrame.

    Note - input InferenceData must be the full (unburnt) idata

    Args:
        idata: The InferenceData to sample
        bcm: The BayesianCompartmentalModel (must be the same BCM used to generate idata)
        n_workers: Number of multiprocessing workers to use; defaults to cpu_count/2

    Returns:
        A DataFrame with index (chain, draw) and columns being the keys in ResultsData.extras
            - Use df.reset_index(level="chain").pivot(columns="chain") to move chain into column multiindex
    """
    n_workers = n_workers or int(cpu_count() / 2)

    accepted_df = idata.sample_stats.accepted.to_dataframe()

    # Get indices and samples of accepted runs only (ie unique valid paramsets)
    accepted_indices = [
        (chain, draw) for (chain, draw), accepted in accepted_df.iterrows() if accepted["accepted"]
    ]
    # accepted_samples_df = idata.posterior.to_dataframe().loc[accepted_indices]

    accept_mask = idata.sample_stats.accepted.data
    posterior_t = idata.posterior.transpose("chain", "draw", ...)

    components = {}
    for dv in posterior_t.data_vars:
        components[dv] = posterior_t[dv].data[accept_mask]

    accepted_si = SampleIterator(components, index=accepted_indices)
    # Get the likelihood extras for all accepted samples - this spins up a multiprocessing pool
    # pres = sample_likelihood_extras_mp(bcm, accepted_samples_df, n_workers)

    extras_df = likelihood_extras_for_samples(accepted_si, bcm, n_workers)

    # Create a DataFrame with the full index of the idata
    # This has a lot of redundant information, but it's still only a few Mb and
    # makes lookup _so_ much easier...
    filled_edf = pd.DataFrame(index=accepted_df.index, columns=extras_df.columns)

    for (chain, draw), accepted_s in accepted_df.iterrows():
        # Extract the bool from the Series
        accepted = accepted_s["accepted"]
        # Update the index if this sample is accepted - otherwise we'll
        # store the previous known good sample (ala MCMC)
        if accepted:
            last_good_sample_idx = (chain, draw)
        filled_edf.loc[(chain, draw)] = extras_df.loc[last_good_sample_idx]

    return filled_edf


def likelihood_extras_for_samples(
    samples_df: pd.DataFrame, bcm: BayesianCompartmentalModel, n_workers: int = None
) -> pd.DataFrame:
    def get_sample_extras(sample_params: Tuple[SampleIndex, ParamDict]) -> Tuple[SampleIndex, Dict]:
        """Run the BCM for a given set of parameters, and return its extras dictionary
        (likelihood, posterior etc)

        Args:
            sample_params: The parameter set to sample (indexed by chain,draw)

        Returns:
            A tuple of SampleIndex and the ResultsData.extras dictionary
        """

        idx, params = sample_params
        res = bcm.run(params, include_extras=True)
        return idx, res.extras

    pres = map_parallel(get_sample_extras, samples_df.iterrows(), n_workers)

    extras_dict = {
        "logposterior": {},
        "logprior": {},
        "loglikelihood": {},
    }

    base_fields = list(extras_dict)

    for idx, res in pres:
        for field in base_fields:
            extras_dict[field][idx] = float(res[field])
        for k, v in res["ll_components"].items():
            extras_dict.setdefault("ll_" + k, {})
            extras_dict["ll_" + k][idx] = float(v)

    extras_df = pd.DataFrame(extras_dict)

    return extras_df


class SampleIterator:
    """A simple container storing dicts of arrays, providing a means to iterate over
    the array items (and returning a dict of the items at each index)
    Designed to be a drop-in replacement for pd.DataFrame.iterrows (but supporting multidimensional arrays)
    """

    def __init__(self, components: dict, index=None):
        self.components = components
        self._calc_component_length()
        if index is None:
            self.index = np.arange(self.clen)
        else:
            assert (
                idxlen := len(index)
            ) == self.clen, f"Index length {idxlen} not equal to component length {self.clen}"
            self.index = index

    def _calc_component_length(self):
        clen = None
        for k, cval in self.components.items():
            if clen is None:
                clen = len(cval)
            else:
                assert len(cval) == clen, f"Length mismatch for {k} ({len(cval)}), should be {clen}"
        self.clen = clen

    def __iter__(self):
        for i in range(self.clen):
            out = {}
            for k, v in self.components.items():
                out[k] = v[i]
            yield out

    def iterrows(self):
        for i in range(self.clen):
            out = {}
            for k, v in self.components.items():
                out[k] = v[i]
            yield self.index[i], out

    def __getitem__(self, idx):
        out = {}
        for k, v in self.components.items():
            out[k] = v[idx]
        return out
