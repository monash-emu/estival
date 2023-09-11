from typing import List, Union

import numpy as np
import pandas as pd
from scipy.stats import qmc

from estival.priors import PriorDict
from estival.sampling import tools as esamptools

from dataclasses import dataclass


@dataclass
class PriorSizeInfo:
    sizes: List[int]
    tot_size: int
    offsets: List[slice]


class SampleTypes:
    INPUT = "input"
    DICT = "dict"
    LIST_OF_DICTS = "list_of_dicts"
    ARRAY = "array"
    PANDAS = "pandas"
    SAMPLEITERATOR = "sample"


def constrain(sample, priors: PriorDict, bounds=0.99):
    # return {k:np.clip(sample[k], *priors[k].bounds(bounds)) for k,v in sample.items()}
    def constrain_op(sample, prior):
        return np.clip(sample, *prior.bounds(bounds))

    return _process_samples_for_priors(sample, priors, constrain_op)


def ppf(sample, priors: PriorDict):
    def ppf_op(sample, prior):
        return prior.ppf(sample)

    return _process_samples_for_priors(sample, priors, ppf_op)


def cdf(sample, priors: PriorDict):
    def cdf_op(sample, prior):
        return prior.cdf(sample)

    return _process_samples_for_priors(sample, priors, cdf_op)


def get_prior_sizeinfo(priors) -> PriorSizeInfo:
    sizes = [p.size for k, p in priors.items()]
    tot_size = sum(sizes)
    offsets = [0] + list(np.cumsum(sizes))
    offset_idx = []
    for i in range(len(offsets) - 1):
        if sizes[i] == 1:
            offset_idx.append(offsets[i])
        else:
            offset_idx.append(slice(offsets[i], offsets[i + 1]))
    # offset_idx = [slice(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]
    return PriorSizeInfo(sizes, tot_size, offset_idx)


def _process_samples_for_priors(sample, priors: PriorDict, op_func):
    size_info = get_prior_sizeinfo(priors)
    psize = size_info.tot_size
    poffsets = size_info.offsets
    if isinstance(sample, np.ndarray):
        shape = sample.shape
        if len(shape) == 1:
            if len(sample) == psize:
                return np.array(
                    [op_func(sample[poffsets[i]], p) for i, p in enumerate(priors.values())]
                )
            else:
                raise ValueError("Input sample must be same size as priors")
        elif len(shape) == 2:
            if shape[1] == psize:
                out_arr = np.empty_like(sample)
                priors_list = list(priors.values())
                for i, pidx in enumerate(poffsets):
                    # for i, psamp_set in enumerate(sample.T):
                    psamp_set = sample[:, pidx]
                    out_arr[:, pidx] = op_func(psamp_set, priors_list[i])
                return out_arr
            else:
                raise ValueError(
                    "Shape mismatch: Could not broadcast input sample to priors", shape
                )
        else:
            raise ValueError(f"Invalid shape {shape} for sample")
    elif isinstance(sample, dict):
        return {k: op_func(v, priors[k]) for k, v in sample.items()}
    elif isinstance(sample, pd.Series):
        return pd.Series({k: op_func(v, priors[k]) for k, v in sample.items()})
    elif isinstance(sample, pd.DataFrame):
        out_df = pd.DataFrame(index=sample.index)
        for c in sample.columns:
            out_df[c] = op_func(sample[c].to_numpy(), priors[c])
        return out_df
    elif isinstance(sample, list):
        assert all([isinstance(subsample, dict) for subsample in sample])
        return [_process_samples_for_priors(subsample, priors, op_func) for subsample in sample]


def convert_sample_type(sample, priors, target_type: str):
    if target_type == SampleTypes.INPUT:
        return sample

    size_info = get_prior_sizeinfo(priors)
    psize = size_info.tot_size

    if target_type == SampleTypes.SAMPLEITERATOR:
        if isinstance(sample, np.ndarray):
            return esamptools.SampleIterator.from_array(sample, priors)
        else:
            return esamptools.validate_samplecontainer(sample)

    if isinstance(sample, np.ndarray):
        if target_type == SampleTypes.ARRAY:
            return sample
        elif target_type == SampleTypes.LIST_OF_DICTS:
            if len(sample.shape) == 1:
                sample = sample.reshape((len(sample), 1))
            if len(sample.shape) == 2:
                return [
                    {k: subsample[size_info.offsets[i]] for i, k in enumerate(priors)}
                    for subsample in sample
                ]
            else:
                raise ValueError(
                    "Shape mismatch: Could not broadcast input sample to priors", sample.shape
                )
        elif target_type == SampleTypes.DICT:
            assert len(sample.shape) == 1
            assert len(sample) == psize
            return {k: sample[i] for i, k in enumerate(priors)}
        elif target_type == SampleTypes.PANDAS:
            df = pd.DataFrame(sample, columns=priors)
            df.index.name = "sample"
            return df
        else:
            raise ValueError(f"Target type {target_type} not supported for array inputs")
    elif isinstance(sample, list):
        assert isinstance(sample[0], dict)
        if target_type == SampleTypes.ARRAY:
            return _lod_to_arr(sample, priors)
        elif target_type == SampleTypes.PANDAS:
            df = pd.DataFrame(_lod_to_arr(sample, priors), columns=priors)
            df.index.name = "sample"
            return df
    elif isinstance(sample, pd.DataFrame):
        if target_type == SampleTypes.LIST_OF_DICTS:
            return [v.to_dict() for _, v in sample.iterrows()]
        elif target_type == SampleTypes.ARRAY:
            return sample.to_numpy()
        elif target_type == SampleTypes.PANDAS:
            return sample
    else:
        sample = esamptools.validate_samplecontainer(sample)
        if target_type == SampleTypes.LIST_OF_DICTS:
            return [v for _, v in sample.iterrows()]  # type: ignore
        elif target_type == SampleTypes.PANDAS:
            return pd.DataFrame(sample.components, index=sample.index)

    raise TypeError(
        "Unsupported combination of input type and target type", type(sample), target_type
    )


def _lod_to_arr(in_lod, priors):
    assert len(in_lod[0]) == len(priors)
    out_arr = np.empty((len(in_lod), len(in_lod[0])))
    for i, in_dict in enumerate(in_lod):
        for j, k in enumerate(priors):
            out_arr[i, j] = in_dict[k]
    return out_arr


def _lod_to_si(lod):
    ref_dict = lod[0]
    components = {}
    for k in ref_dict:
        components[k] = np.stack([samp[k] for samp in lod])
    return esamptools.SampleIterator(components)


class SampledPriorsManager:
    def __init__(self, priors):
        self.priors = priors
        self.size_info = get_prior_sizeinfo(priors)

    def constrain(self, sample, bounds=0.99, ret_type=SampleTypes.INPUT):
        return convert_sample_type(constrain(sample, self.priors, bounds), self.priors, ret_type)

    def ppf(self, sample, ret_type=SampleTypes.INPUT):
        return convert_sample_type(ppf(sample, self.priors), self.priors, ret_type)

    def cdf(self, sample, ret_type=SampleTypes.INPUT):
        return convert_sample_type(cdf(sample, self.priors), self.priors, ret_type)

    def convert(self, sample, ret_type):
        return convert_sample_type(sample, self.priors, ret_type)

    def lhs(self, n_samples: int, out_type="sample", clamp=0.99):
        lhs = qmc.LatinHypercube(self.size_info.tot_size)
        samples = lhs.random(n_samples)
        clamp_offset = (1.0 - clamp) * 0.5
        samples = np.clip(samples, clamp_offset, 1.0 - clamp_offset)
        resampled = self.ppf(samples)
        return self.convert(resampled, out_type)

    def sobol(self, n_samples: int, out_type="sample", clamp=0.99):
        sobol = qmc.Sobol(self.size_info.tot_size)
        samples = sobol.random(n_samples)
        clamp_offset = (1.0 - clamp) * 0.5
        samples = np.clip(samples, clamp_offset, 1.0 - clamp_offset)
        resampled = self.ppf(samples)
        return self.convert(resampled, out_type)

    def uniform(self, n_samples: int, out_type="sample", clamp=0.99):
        samples = np.random.uniform(size=(n_samples, self.size_info.tot_size))
        clamp_offset = (1.0 - clamp) * 0.5
        samples = np.clip(samples, clamp_offset, 1.0 - clamp_offset)
        resampled = self.ppf(samples)
        return self.convert(resampled, out_type)
