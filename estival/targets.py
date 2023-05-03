from __future__ import annotations

from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from copy import copy

import pandas as pd
import numpy as np

from jax import jit, scipy as jsp, numpy as jnp

from .priors import DistriParam, BasePrior


class BaseTarget(ABC):
    name: str
    data: pd.Series
    _data_attrs: List = []

    def __init__(
        self, name: str, data: pd.Series, weight: float = 1.0, time_weights: pd.Series = None
    ):
        # Make things easier for calibration by sanitizing the data here
        self.name = name
        self.data = data

        # Should do some validation on this - ie make sure indices match data
        # if time_weights is None:
        #    time_weights = pd.Series(index=data.index, data=np.repeat(1.0 / len(data), len(data)))

        self.time_weights = time_weights
        self.weight = weight

    def get_priors(self):
        return []

    def filtered(self, index: pd.Index) -> BaseTarget:
        out_target = copy(self)
        valid_idx = index.intersection(self.data.index)
        out_target.data = out_target.data[valid_idx]

        for da in self._data_attrs:
            data = getattr(out_target, da)
            setattr(out_target, da, data[valid_idx])

        if self.time_weights is not None:
            new_time_weights = out_target.time_weights[valid_idx]
            out_target.time_weights = new_time_weights / new_time_weights.sum()
        return out_target

    @abstractmethod
    def get_evaluator(self, model_times: pd.Index) -> TargetEvaluator:
        raise NotImplementedError()


class TargetEvaluator(ABC):
    def __init__(self, target: BaseTarget, model_times: pd.Index):
        self.target = target.filtered(model_times)
        self.data = self.target.data.to_numpy()
        for da in target._data_attrs:
            data = getattr(self.target, da)
            setattr(self, da, data.to_numpy())
        self.index = np.array([model_times.get_loc(t) for t in self.target.data.index])
        if self.target.time_weights is not None:
            self.time_weights = self.target.time_weights.to_numpy()
        else:
            self.time_weights = None

    @abstractmethod
    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        raise NotImplementedError()


class NegativeBinomialEvaluator(TargetEvaluator):
    def __init__(self, target: BaseTarget, model_times: pd.Index):
        super().__init__(target, model_times)

    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        if isinstance(self.target.dispersion_param, BasePrior):
            n = parameters[self.target.dispersion_param.name]
        else:
            n = self.target.dispersion_param

        # We use the parameterisation based on mean and variance and assume define var=mean**delta
        mu = modelled[self.index]
        # work out parameter p to match the distribution mean with the model output
        p = mu / (mu + n)
        # Attempt to minimize -inf showing up
        p = jnp.where(p == 0.0, 1e-16, p)
        # ll = np.sum(stats.nbinom.logpmf(self.data, n, 1.0 - p) * self.time_weights)
        ll = jsp.stats.nbinom.logpmf(self.data, n, 1.0 - p)

        if self.time_weights is not None:
            ll = ll * self.time_weights
            return jnp.sum(ll) * self.target.weight
        else:
            return jnp.mean(ll) * self.target.weight


class NegativeBinomialTarget(BaseTarget):
    """
    A calibration target sampled from a negative binomial distribution
    """

    def __init__(
        self,
        name: str,
        data: pd.Series,
        dispersion_param: DistriParam,
        weight: float = 1.0,
        time_weights: pd.Series = None,
    ):
        super().__init__(name, data, weight, time_weights)
        self.dispersion_param = dispersion_param

    def get_priors(self):
        if isinstance(self.dispersion_param, BasePrior):
            return [self.dispersion_param]
        else:
            return []

    def get_evaluator(self, model_times: pd.Index) -> TargetEvaluator:
        return NegativeBinomialEvaluator(self, model_times)


class BinomialTarget(BaseTarget):
    """
    A calibration target sampled from a binomial distribution
    """

    def __init__(
        self,
        name: str,
        data: pd.Series,
        sample_sizes: pd.Series,
        weight: float = 1.0,
        time_weights: pd.Series = None,
    ):
        super().__init__(name, data, weight, time_weights)
        self._data_attrs = ["sample_sizes"]
        self.sample_sizes = sample_sizes

    def get_evaluator(self, model_times: pd.Index) -> TargetEvaluator:
        return BinomialEvaluator(self, model_times)


class BinomialEvaluator(TargetEvaluator):
    def __init__(self, target: BaseTarget, model_times: pd.Index):
        super().__init__(target, model_times)
        # Enforce this here so TFP-jax picks the right output types
        self.sample_sizes = self.sample_sizes.astype(float)

    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        from tensorflow_probability.substrates import jax as tfp

        # use a binomial (n, p) where n is the sample size observed in the data and p the modelled proportion
        # We then evaluate the binomial density for k, which represents the numerator observed in the data
        n = self.target.sample_sizes
        p = modelled
        k = self.target.data * n

        bdist = tfp.distributions.Binomial(total_count=n, probs=p)
        ll = bdist.log_prob(k)

        if self.time_weights is not None:
            ll = ll * self.time_weights
            return jnp.sum(ll) * self.target.weight
        else:
            return jnp.mean(ll) * self.target.weight


class TruncatedNormalTarget(BaseTarget):
    """
    A calibration target sampled from a truncated normal distribution
    """

    def __init__(
        self,
        name: str,
        data: pd.Series,
        trunc_range: Tuple[float, float],
        stdev: DistriParam,
        weight: float = 1.0,
        time_weights: pd.Series = None,
    ):
        super().__init__(name, data, weight, time_weights)
        self.trunc_range = trunc_range
        self.stdev = stdev

    def get_priors(self):
        if isinstance(self.stdev, BasePrior):
            return [self.stdev]
        else:
            return []

    def get_evaluator(self, model_times: pd.Index) -> TargetEvaluator:
        return TruncatedNormalTargetEvaluator(self, model_times)


class TruncatedNormalTargetEvaluator(TargetEvaluator):
    def __init__(self, target: TruncatedNormalTarget, model_times: pd.Index):
        super().__init__(target, model_times)

    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        if isinstance(self.target.stdev, BasePrior):
            sd = parameters[self.target.stdev.name]
        else:
            sd = self.target.stdev

        distri_params = {
            "scale": sd,
            "a": (self.target.trunc_range[0] - self.data) / sd,
            "b": (self.target.trunc_range[1] - self.data) / sd,
        }

        ll = jsp.stats.truncnorm.logpdf(modelled[self.index], loc=self.data, **distri_params)

        if self.time_weights is not None:
            ll = ll * self.time_weights
            return jnp.sum(ll) * self.target.weight
        else:
            return jnp.mean(ll) * self.target.weight


class NormalTarget(BaseTarget):
    """
    A calibration target sampled from a normal distribution
    """

    def __init__(
        self,
        name: str,
        data: pd.Series,
        stdev: DistriParam,
        weight: float = 1.0,
        time_weights: pd.Series = None,
    ):
        super().__init__(name, data, weight, time_weights)
        self.stdev = stdev

    def get_priors(self):
        if isinstance(self.stdev, BasePrior):
            return [self.stdev]
        else:
            return []

    def get_evaluator(self, model_times: pd.Index) -> TargetEvaluator:
        return NormalTargetEvaluator(self, model_times)


class NormalTargetEvaluator(TargetEvaluator):
    def __init__(self, target: BaseTarget, model_times: pd.Index):
        super().__init__(target, model_times)

    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        if isinstance(self.target.stdev, BasePrior):
            sd = parameters[self.target.stdev.name]
        else:
            sd = self.target.stdev

        ll = jsp.stats.norm.logpdf(modelled[self.index], loc=self.data, scale=sd)

        if self.time_weights is not None:
            ll = ll * self.time_weights
            return jnp.sum(ll) * self.target.weight
        else:
            return jnp.mean(ll) * self.target.weight


class CustomTargetEvaluator(TargetEvaluator):
    def __init__(self, target, model_times, eval_func):
        super().__init__(target, model_times)
        self._eval_func = eval_func

    def evaluate(self, modelled, parameters):
        return (
            self._eval_func(modelled[self.index], self.data, parameters, self.time_weights)
            * self.target.weight
        )


class CustomTarget(BaseTarget):
    def __init__(self, name, data, eval_func, weight=1.0, time_weights=None):
        """Build a Target that uses a custom evaluation function.
        Indexing and multiplication by weight factor is handled automatically,
        but time_weights are not (see below)

        For example:

        def least_squares(modelled, obs, parameters, time_weights):
            return (((modelled - obs) ** 2.0) * time_weights).sum()

        CustomTarget("example", example_data, least_squares)

        Args:
            name: Name (key) of output to evaluate against
            data: Series of data using same indexing conventions as model
            eval_func: Callable as described above
            weight (optional): Scales resulting output
            time_weights (optional): Passed to eval_func - Series with index matching data
        """
        super().__init__(name, data, weight, time_weights)
        self.eval_func = eval_func

    def get_evaluator(self, model_times):
        return CustomTargetEvaluator(self, model_times, self.eval_func)


def get_target_sd(data: pd.Series) -> float:
    """Return a value such that the 95% CI of the associated normal distribution covers a width
       equivalent to 25% of the maximum value of the target.

    Args:
        data: The target data series

    Returns:
        Calculated sd
    """
    return 0.25 / 4.0 * max(data)
