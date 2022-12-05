from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from copy import copy

import pandas as pd
import numpy as np
from scipy import stats

from jax import jit, scipy as jsp, numpy as jnp

from .priors import DistriParam, BasePrior


class TargetEvaluator(ABC):
    @abstractmethod
    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        raise NotImplementedError()


class BaseTarget(ABC):

    name: str
    data: pd.Series

    def __init__(self, name: str, data: pd.Series, time_weights: pd.Series = None):
        # Make things easier for calibration by sanitizing the data here
        self.name = name
        self.data = data

        # Should do some validation on this - ie make sure indices match data
        if time_weights is None:
            time_weights = pd.Series(index=data.index, data=np.repeat(1.0 / len(data), len(data)))

        self.time_weights = time_weights

    def get_priors(self):
        return []

    def filtered(self, index: pd.Index):
        out_target = copy(self)
        valid_idx = index.intersection(self.data.index)
        out_target.data = out_target.data[valid_idx]
        new_time_weights = out_target.time_weights[valid_idx]
        out_target.time_weights = new_time_weights / new_time_weights.sum()
        return out_target

    @abstractmethod
    def get_evaluator(self, model_times: pd.Index) -> TargetEvaluator:
        raise NotImplementedError()


def _build_jax_eval_nbinom(evaluator):
    @jit
    def _jax_evaluate_nbinom(modelled: np.array, parameters: dict) -> float:
        if isinstance(evaluator.target.dispersion_param, BasePrior):
            n = parameters[evaluator.target.dispersion_param.name]
        else:
            n = evaluator.target.dispersion_param

        # We use the parameterisation based on mean and variance and assume define var=mean**delta
        mu = modelled
        # work out parameter p to match the distribution mean with the model output
        p = mu / (mu + n)
        # Attempt to minimize -inf showing up
        p = jnp.where(p == 0.0, 1e-16, p)
        # ll = np.sum(stats.nbinom.logpmf(self.data, n, 1.0 - p) * self.time_weights)
        ll = jnp.sum(jsp.stats.nbinom.logpmf(evaluator.data, n, 1.0 - p) * evaluator.time_weights)
        return ll

    return _jax_evaluate_nbinom


class NegativeBinomialEvaluator(TargetEvaluator):
    def __init__(self, target: BaseTarget, model_times: pd.Index):
        self.target = target.filtered(model_times)
        self.data = self.target.data.round().to_numpy()
        self.index = np.array([model_times.get_loc(t) for t in self.target.data.index])
        self.time_weights = self.target.time_weights.to_numpy()
        self._eval_func = _build_jax_eval_nbinom(self)

    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        return float(self._eval_func(modelled[self.index], parameters))


class NegativeBinomialTarget(BaseTarget):
    """
    A calibration target sampled from a negative binomial distribution
    """

    def __init__(
        self,
        name: str,
        data: pd.Series,
        dispersion_param: DistriParam,
        time_weights: pd.Series = None,
    ):
        super().__init__(name, data, time_weights)
        self.dispersion_param = dispersion_param

    def get_priors(self):
        if isinstance(self.dispersion_param, BasePrior):
            return [self.dispersion_param]
        else:
            return []

    def get_evaluator(self, model_times: pd.Index) -> TargetEvaluator:
        return NegativeBinomialEvaluator(self, model_times)


class TruncNormalTarget(BaseTarget):
    """
    A calibration target sampled from a truncated normal distribution
    """

    def __init__(
        self,
        name: str,
        data: pd.Series,
        trunc_range: Tuple[float, float],
        stdev: DistriParam,
        time_weights: pd.Series = None,
    ):
        super().__init__(name, data, time_weights)
        self.trunc_range = trunc_range
        self.stdev = stdev

    def get_priors(self):
        if isinstance(self.stdev, BasePrior):
            return [self.stdev]
        else:
            return []


class NormalTarget(BaseTarget):
    """
    A calibration target sampled from a normal distribution
    """

    def __init__(
        self, name: str, data: pd.Series, stdev: DistriParam, time_weights: pd.Series = None
    ):
        super().__init__(name, data, time_weights)
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
        self.target = target.filtered(model_times)
        self.data = self.target.data.to_numpy()
        self.index = np.array([model_times.get_loc(t) for t in self.target.data.index])
        self.time_weights = self.target.time_weights.to_numpy()
        self._eval_func = _build_jax_eval_nbinom(self)

    def evaluate(self, modelled: np.array, parameters: dict) -> float:
        if isinstance(self.target.stdev, BasePrior):
            sd = parameters[self.target.stdev.name]
        else:
            sd = self.target.stdev
        return (
            stats.norm.logpdf(modelled[self.index], loc=self.data, scale=sd) * self.time_weights
        ).sum()


def get_target_sd(data: pd.Series) -> float:
    """Return a value such that the 95% CI of the associated normal distribution covers a width
       equivalent to 25% of the maximum value of the target.

    Args:
        data: The target data series

    Returns:
        Calculated sd
    """
    return 0.25 / 4.0 * max(data)
