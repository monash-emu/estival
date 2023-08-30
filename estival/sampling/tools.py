from typing import Tuple, Dict, Optional, Union
from multiprocessing import cpu_count

import pandas as pd
from arviz import InferenceData
import numpy as np

import xarray

from estival.model import BayesianCompartmentalModel, ResultsData
from estival.utils.parallel import map_parallel

SampleIndex = Tuple[int, int]
ParamDict = Dict[str, float]
SampleContainer = Union[pd.DataFrame, "SampleIterator", xarray.Dataset]


def likelihood_extras_for_idata(
    idata: InferenceData, bcm: BayesianCompartmentalModel, n_workers: Optional[int] = None
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

    accepted_s = idata["sample_stats"].accepted.copy()
    # Handle pathological cases where we've burnt out the first accepted sample
    accepted_s[:, 0] = True

    accepted_df = accepted_s.to_dataframe()

    # Get indices and samples of accepted runs only (ie unique valid paramsets)
    accepted_indices = [
        (chain, draw) for (chain, draw), accepted in accepted_df.iterrows() if accepted["accepted"]
    ]
    # accepted_samples_df = idata.posterior.to_dataframe().loc[accepted_indices]

    accept_mask = accepted_s.data
    posterior_t = idata["posterior"].transpose("chain", "draw", ...)

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


def _extras_df_from_pres(pres, is_full_data=False) -> pd.DataFrame:
    extras_dict = {
        "logposterior": {},
        "logprior": {},
        "loglikelihood": {},
    }

    base_fields = list(extras_dict)

    for idx, res in pres:
        if is_full_data:
            extras = res.extras
        else:
            extras = res
        for field in base_fields:
            extras_dict[field][idx] = float(extras[field])
        for k, v in extras["ll_components"].items():
            extras_dict.setdefault("ll_" + k, {})
            extras_dict["ll_" + k][idx] = float(v)

    extras_df = pd.DataFrame(extras_dict)
    extras_df.index = extras_df.index.set_names(("chain", "draw"))  # pyright: ignore

    return extras_df


def likelihood_extras_for_samples(
    samples: SampleContainer, bcm: BayesianCompartmentalModel, n_workers: Optional[int] = None
) -> pd.DataFrame:
    def get_sample_extras(sample_params: Tuple[SampleIndex, ParamDict]) -> Tuple[SampleIndex, dict]:
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

    samples = validate_samplecontainer(samples)

    pres = map_parallel(get_sample_extras, samples.iterrows(), n_workers)

    return _extras_df_from_pres(pres, False)


def model_results_for_samples(
    samples: SampleContainer,
    bcm: BayesianCompartmentalModel,
    include_extras: bool = False,
    n_workers: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    def get_model_results(
        sample_params: Tuple[SampleIndex, ParamDict]
    ) -> Tuple[SampleIndex, ResultsData]:
        """Run the BCM for a given set of parameters, and return its extras dictionary
        (likelihood, posterior etc)

        Args:
            sample_params: The parameter set to sample (indexed by chain,draw)

        Returns:
            A tuple of SampleIndex and the ResultsData.extras dictionary
        """

        idx, params = sample_params
        res = bcm.run(params, include_extras=include_extras)
        return idx, res

    samples = validate_samplecontainer(samples)

    pres = map_parallel(get_model_results, samples.iterrows(), n_workers)

    df = pd.concat([p[1].derived_outputs for p in pres], keys=[p[0] for p in pres])
    df: pd.DataFrame = df.sort_index().unstack(level=(0, 1))  # type: ignore
    if include_extras:
        extras_df: pd.DataFrame = _extras_df_from_pres(pres, True).sort_index()  # type:ignore
        return df, extras_df
    else:
        return df


class SampleIterator:
    """A simple container storing dicts of arrays, providing a means to iterate over
    the array items (and returning a dict of the items at each index)
    Designed to be a drop-in replacement for pd.DataFrame.iterrows (but supporting multidimensional arrays)
    """

    def __init__(self, components: dict, index=None):
        self.components = components
        self.clen = self._calc_component_length()
        if index is None:
            self.index = np.arange(self.clen)
        else:
            assert (
                idxlen := len(index)
            ) == self.clen, f"Index length {idxlen} not equal to component length {self.clen}"
            self.index = index

    def _calc_component_length(self) -> int:
        clen = -1
        for k, cval in self.components.items():
            if clen == -1:
                clen = len(cval)
            else:
                assert len(cval) == clen, f"Length mismatch for {k} ({len(cval)}), should be {clen}"
        return clen

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


def xarray_to_sampleiterator(in_data: xarray.Dataset):
    if list(in_data.dims) == ["sample"]:
        index = in_data.sample.to_index()
        data_t = in_data.transpose("sample", ...)
    elif list(in_data.dims) == ["chain", "draw"]:
        index = in_data.coords.to_index()
        data_t = in_data.transpose("chain", "draw", ...)
    else:
        raise KeyError("Incompatible dimensions ")

    components = {}
    for dv in in_data.data_vars:
        components[dv] = data_t[dv].data

    si = SampleIterator(components, index=index)
    return si


def idata_to_sampleiterator(in_data: InferenceData, group="posterior"):
    return xarray_to_sampleiterator(in_data[group])


def validate_samplecontainer(in_data: SampleContainer) -> Union[SampleIterator, pd.DataFrame]:
    if isinstance(in_data, InferenceData):
        return idata_to_sampleiterator(in_data)
    elif isinstance(in_data, xarray.Dataset):
        return xarray_to_sampleiterator(in_data)
    elif isinstance(in_data, SampleIterator):
        return in_data
    elif isinstance(in_data, pd.DataFrame):
        return in_data
    else:
        raise TypeError("Unsupported type", in_data)
