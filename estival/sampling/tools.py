from inspect import istraceback
from typing import Tuple, Dict, Optional, Union
from multiprocessing import cpu_count
from dataclasses import dataclass

import pandas as pd
from arviz import InferenceData
import numpy as np
from pandas.core.frame import dataclasses_to_dicts

import xarray

from estival.model import BayesianCompartmentalModel, ResultsData
from estival.utils.parallel import map_parallel
from estival.utils.sample import convert_sample_type, get_prior_sizeinfo, _lod_to_si

SampleIndex = Tuple[int, int]
ParamDict = Dict[str, float]
SampleContainer = Union[pd.DataFrame, "SampleIterator", xarray.Dataset]


@dataclass
class SampledResults:
    results: pd.DataFrame
    extras: Optional[pd.DataFrame]


@dataclass
class _PriorStub:
    size: int


def likelihood_extras_for_idata(
    idata: InferenceData,
    bcm: BayesianCompartmentalModel,
    num_workers: Optional[int] = None,
    exec_mode: str = "thread",
) -> pd.DataFrame:
    """Calculate the likelihood extras (ll,lprior,lpost + per-target) for all
    samples in supplied InferenceData, returning a DataFrame.

    Note - input InferenceData must be the full (unburnt) idata

    Args:
        idata: The InferenceData to sample
        bcm: The BayesianCompartmentalModel (must be the same BCM used to generate idata)
        num_workers: Number of multiprocessing workers to use; defaults to cpu_count/2

    Returns:
        A DataFrame with index (chain, draw) and columns being the keys in ResultsData.extras
            - Use df.reset_index(level="chain").pivot(columns="chain") to move chain into column multiindex
    """
    num_workers = num_workers or int(cpu_count() / 2)

    accepted_s = idata["sample_stats"].accepted.copy()
    # Handle pathological cases where we've burnt out the first accepted sample
    accepted_s[:, 0] = True

    accepted_df = accepted_s.to_dataframe()

    accepted_index = accepted_df[accepted_df["accepted"] == True].index

    accept_mask = accepted_s.data
    posterior_t = idata["posterior"].transpose("chain", "draw", ...)

    components = {}
    for dv in posterior_t.data_vars:
        components[dv] = posterior_t[dv].data[accept_mask]

    accepted_si = SampleIterator(components, index=accepted_index)
    # Get the likelihood extras for all accepted samples - this spins up a multiprocessing pool
    # pres = sample_likelihood_extras_mp(bcm, accepted_samples_df, n_workers)

    extras_df = likelihood_extras_for_samples(accepted_si, bcm, num_workers, exec_mode=exec_mode)

    # Collate this into an array - it's much much faster than dealing with pandas directly
    tmp_extras = np.empty((len(accepted_df), extras_df.shape[-1]))

    # This value should never get used - we know something went wrong if the accepted field if it did
    last_good_sample_idx = "IndexNotSet"

    for i, (idx, accepted_s) in enumerate(accepted_df.iterrows()):
        # Extract the bool from the Series
        accepted = accepted_s["accepted"]
        # Update the index if this sample is accepted - otherwise we'll
        # store the previous known good sample (ala MCMC)
        if accepted:
            last_good_sample_idx = idx
        tmp_extras[i] = extras_df.loc[last_good_sample_idx]

    # Create a DataFrame with the full index of the idata
    # This has a lot of redundant information, but it's still only a few Mb and
    # makes lookup _so_ much easier...
    filled_edf = pd.DataFrame(
        index=accepted_df.index, columns=extras_df.columns, data=tmp_extras, dtype=float
    )

    return filled_edf


def _extras_df_from_pres(pres, is_full_data=False, index_names=("chain", "draw")) -> pd.DataFrame:
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
    extras_df.index = extras_df.index.set_names(index_names)  # pyright: ignore

    return extras_df


def likelihood_extras_for_samples(
    samples: SampleContainer,
    bcm: BayesianCompartmentalModel,
    num_workers: Optional[int] = None,
    exec_mode: Optional[str] = "thread",
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

    pres = map_parallel(get_sample_extras, samples.iterrows(), num_workers, mode=exec_mode)

    if isinstance(samples.index, pd.MultiIndex):
        levels = samples.index.names
        unstack_levels = list(range(len(levels)))
    else:
        unstack_levels = (0,)
        if hasattr(samples.index, "name"):
            name = samples.index.name or "sample"  # type: ignore
        else:
            name = "sample"
        levels = (name,)

    return _extras_df_from_pres(pres, False, index_names=levels)


def model_results_for_samples(
    samples: SampleContainer,
    bcm: BayesianCompartmentalModel,
    include_extras: bool = True,
    num_workers: Optional[int] = None,
    exec_mode: Optional[str] = "thread",
) -> SampledResults:
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

    pres = map_parallel(get_model_results, samples.iterrows(), num_workers, mode=exec_mode)

    df = pd.concat([p[1].derived_outputs for p in pres], keys=[p[0] for p in pres])

    if isinstance(samples.index, pd.MultiIndex):
        levels = samples.index.names
        unstack_levels = list(range(len(levels)))
    else:
        unstack_levels = (0,)
        if hasattr(samples.index, "name"):
            name = samples.index.name or "sample"  # type: ignore
        else:
            name = "sample"
        levels = (name,)

    df: pd.DataFrame = df.sort_index().unstack(level=unstack_levels)  # type: ignore
    df.columns.set_names(["variable", *levels], inplace=True)
    df.index.set_names("time", inplace=True)

    if include_extras:
        extras_df: pd.DataFrame = _extras_df_from_pres(
            pres, True, index_names=levels
        ).sort_index()  # type:ignore
        return SampledResults(df, extras_df)
    else:
        return SampledResults(df, None)


def quantiles_for_results(results_df: pd.DataFrame, quantiles: Tuple[float]) -> pd.DataFrame:
    """Summary

    Args:
        results_df: DataFrame with layout equivalent to model_results_for_samples output
        quantiles: Quantiles to compute [0.0,1.0]

    Returns:
        pd.DataFrame: DataFrame with time as index and [variable, quantile] as columns
    """
    columns = pd.MultiIndex.from_product(
        (results_df.columns.levels[0], quantiles), names=["variable", "quantile"]  # type: ignore
    )
    udf = pd.DataFrame(index=results_df.index, columns=columns)

    for variable in results_df.columns.levels[0]:  # type: ignore
        udf[variable] = results_df[variable].quantile(quantiles, axis=1).T  # type: ignore

    return udf


class IndexGetter:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, key):
        return self.func(key)


class SampleIterator:
    """A simple container storing dicts of arrays, providing a means to iterate over
    the array items (and returning a dict of the items at each index)
    Designed to be a drop-in replacement for pd.DataFrame.iterrows (but supporting multidimensional arrays)
    """

    def __init__(self, components: dict, index=None):
        self.components = components
        self.clen = self._calc_component_length()
        if index is None:
            self.index = pd.RangeIndex(self.clen, name="sample")
        else:
            assert (
                idxlen := len(index)
            ) == self.clen, f"Index length {idxlen} not equal to component length {self.clen}"
            self.index = index
        self._idxidx = pd.Series(index=self.index, data=range(len(self.index)))

        self.iloc = IndexGetter(self._get_by_iloc)

        self._priors_stub = self._build_priorsize_table()

    def _calc_component_length(self) -> int:
        clen = -1
        for k, cval in self.components.items():
            if clen == -1:
                clen = len(cval)
            else:
                assert len(cval) == clen, f"Length mismatch for {k} ({len(cval)}), should be {clen}"
        return clen

    def _build_priorsize_table(self) -> dict:
        """
        A dictionary of stub objects resembling BasePrior that contain only a size field
        """

        priors_stub = {}
        for k, v in self.components.items():
            refv = v[0]
            if isinstance(refv, np.ndarray):
                size = len(refv)
            else:
                size = 1
            priors_stub[k] = _PriorStub(size)
        return priors_stub

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
        if isinstance(idx, pd.Index):
            data_idx = self._idxidx[idx]
            for k, v in self.components.items():
                out[k] = v[data_idx]
            return SampleIterator(out, index=idx)
        else:
            for k, v in self.components.items():
                out[k] = v[idx]
            return out

    def _get_by_iloc(self, key):
        return self[self.index[key]]

    def convert(self, target_type: str):
        return convert_sample_type(self, self._priors_stub, target_type=target_type)

    @classmethod
    def from_array(cls, in_array, priors):
        size_info = get_prior_sizeinfo(priors)
        components = {k: in_array[:, size_info.offsets[i]] for i, k in enumerate(priors)}
        return cls(components)


def xarray_to_sampleiterator(in_data: xarray.Dataset):
    if list(in_data.dims)[0] == "sample":
        index = in_data.sample.to_index()
        data_t = in_data.transpose("sample", ...)
        n_idxdim = 1
    elif list(in_data.dims)[:2] == ["chain", "draw"]:
        index = pd.MultiIndex.from_product(
            [in_data.coords["chain"].to_index(), in_data.coords["draw"].to_index()]
        )
        data_t = in_data.transpose("chain", "draw", ...)
        n_idxdim = 2
    else:
        raise KeyError("Incompatible dimensions ")

    components = {}
    for dv in in_data.data_vars:
        dvar = data_t[dv]

        sample_shape = int(np.prod(dvar.data.shape[:n_idxdim]))
        var_shape = dvar.data.shape[n_idxdim:]
        final_shape = [sample_shape] + list(var_shape)

        components[dv] = dvar.data.reshape(final_shape)

    si = SampleIterator(components, index=index)
    return si


def idata_to_sampleiterator(in_data: InferenceData, group="posterior"):
    return xarray_to_sampleiterator(in_data[group])


def dataframe_to_sampleiterator(in_data: pd.DataFrame):
    components = {c: in_data[c].to_numpy() for c in in_data.columns}  # type: ignore
    return SampleIterator(components, index=in_data.index)


def validate_samplecontainer(in_data: SampleContainer) -> Union[SampleIterator, pd.DataFrame]:
    if isinstance(in_data, SampledResults):
        return in_data.results
    if isinstance(in_data, InferenceData):
        return idata_to_sampleiterator(in_data)
    elif isinstance(in_data, xarray.Dataset):
        return xarray_to_sampleiterator(in_data)
    elif isinstance(in_data, SampleIterator):
        return in_data
    elif isinstance(in_data, pd.DataFrame):
        return dataframe_to_sampleiterator(in_data)
    elif isinstance(in_data, list):
        return _lod_to_si(in_data)
    else:
        raise TypeError("Unsupported type", in_data)
