from typing import Any, Tuple, Union, Dict, cast
from abc import ABC

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pandas as pd

# pymc is optional - just be silent on failed import
try:
    import pymc as pm
except:
    pass


class BasePrior(ABC):
    _rv: stats.distributions.rv_frozen

    def __init__(self, name: str, size: int = 1):
        self.name = name
        self.size = size

    def bounds(self, ci=1.0) -> Tuple[float, float]:
        return self._rv.interval(ci)

    def finite_bounds(self, ci=0.99) -> Tuple[float, float]:
        """Return guaranteed finite bounds, each of the lower and upper bounds
        will be either its full extremum (if finite), or the bound specified by
        the supplied confidence interval

        Args:
            ci: Confidence interval for alternate bounds

        Returns:
            Tuple of lower, upper
        """
        lower, upper = self.bounds(1.0)
        cbounds = self.bounds(ci)
        if np.isinf(lower):
            lower = cbounds[0]
        if np.isinf(upper):
            upper = cbounds[1]
        return lower, upper

    def ppf(self, q):
        """Probability Percentage Function at q (Inverse CDF)
        Defaults to using the underlying scipy distribution function

        Args:
            q: Quantile (float or arraylike) at which to evaluate ppf

        Returns:
            typeof(q): The ppf values
        """

        return self._rv.ppf(q)

    def cdf(self, x):
        """Cumulative Distribution Function at x
        Defaults to using the underlying scipy distribution function

        Args:
            x: Value (float or arraylike) at which to evaluate cdf

        Returns:
            typeof(x): The cdf values
        """
        return self._rv.cdf(x)

    def pdf(self, x):
        """Probability Density Function at x
        Defaults to using the underlying scipy distribution function

        Args:
            x: Value (float or arraylike) at which to evaluate pdf

        Returns:
            typeof(x): The pdf values
        """
        return self._rv.pdf(x)

    def logpdf(self, x):
        """Log Probability Density Function at x
        Defaults to using the underlying scipy distribution function

        Args:
            x: Value (float or arraylike) at which to evaluate logpdf

        Returns:
            typeof(x): The logpdf values
        """
        return self._rv.logpdf(x)

    def get_series(self, func_name, ci=0.99, slen=101):
        x = np.linspace(*self.finite_bounds(ci=ci), slen)
        y = getattr(self._rv, func_name)(x)
        return pd.Series(y, x, name=func_name)

    def _get_pymc_shape(self):
        if self.size == 1:
            return None
        else:
            return self.size

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}"

    def to_pymc(self):
        raise NotImplementedError()

    @classmethod
    def _get_test(cls):
        raise NotImplementedError()


# Union type for hierarchical/dispersion parameters
DistriParam = Union[float, BasePrior]

# Dict type used by BayesianCompartmentalModel
PriorDict = Dict[str, BasePrior]


class BetaPrior(BasePrior):
    """
    A beta distributed prior.
    """

    def __init__(self, name: str, a: float, b: float, size=1):
        super().__init__(name, size)
        self.a = float(a)
        self.b = float(b)
        self.distri_params = {"a": a, "b": b}
        self._rv = stats.beta(a, b)

    @classmethod
    def from_mean_and_ci(
        cls, name: str, mean: float, ci: Tuple[float, float], ci_width=0.95, size=1
    ):
        assert len(ci) == 2 and ci[1] > ci[0] and 0.0 < ci_width < 1.0
        percentile_low = (1.0 - ci_width) / 2.0
        percentile_up = 1.0 - percentile_low
        assert 0.0 < ci[0] < 1.0 and 0.0 < ci[1] < 1.0 and 0.0 < mean < 1.0

        def distance_to_minimise(a):
            b = a * (1.0 - mean) / mean
            vals = stats.beta.ppf([percentile_low, percentile_up], a, b)
            dist = sum([(ci[i] - vals[i]) ** 2 for i in range(2)])
            return dist

        sol = minimize(distance_to_minimise, [1.0], bounds=[(0.0, None)], tol=1.0e-32)
        best_a = sol.x
        best_b = best_a * (1.0 - mean) / mean

        return cls(name, best_a[0], best_b[0], size)

    def to_pymc(self):
        return pm.Beta(self.name, alpha=self.a, beta=self.b, shape=self._get_pymc_shape())

    @classmethod
    def _get_test(cls):
        return cls("test", 2.0, 5.0)


class UniformPrior(BasePrior):
    """
    A uniformily distributed prior.
    """

    def __init__(self, name: str, domain: Tuple[float, float], size=1):
        super().__init__(name)
        self.start, self.end = domain
        self.distri_params = {"loc": self.start, "scale": self.end - self.start}
        self._rv = stats.uniform(**self.distri_params)
        self.size = size
        self._pymc_transform_eps_scale = 0.0

    def to_pymc(self):
        lower, upper = self.start, self.end
        eps = self._pymc_transform_eps_scale * (upper - lower)

        if self._pymc_transform_eps_scale != 0.0:
            interval_transform = pm.distributions.transforms.Interval(
                lower=lower - eps, upper=upper + eps
            )
            return pm.Uniform(
                self.name,
                lower=lower,
                upper=upper,
                transform=interval_transform,
                shape=self._get_pymc_shape(),
            )
        else:
            return pm.Uniform(
                self.name,
                lower=lower,
                upper=upper,
                shape=self._get_pymc_shape(),
            )

    def __repr__(self):
        return f"{super().__repr__()} {{bounds: {self.bounds()}}}"

    @classmethod
    def _get_test(cls):
        return cls("test", (0.0, 1.0))


class TruncNormalPrior(BasePrior):
    """
    A prior with a truncated normal distribution.
    """

    def __init__(
        self, name: str, mean: float, stdev: float, trunc_range: Tuple[float, float], size=1
    ):
        super().__init__(name, size)
        self.mean, self.stdev = mean, stdev
        self.trunc_range = tuple(trunc_range)
        self.distri_params = {
            "loc": mean,
            "scale": stdev,
            "a": (trunc_range[0] - mean) / stdev,
            "b": (trunc_range[1] - mean) / stdev,
        }
        self._rv = stats.truncnorm(**self.distri_params)

    def to_pymc(self):
        lower, upper = self.trunc_range
        return pm.TruncatedNormal(
            self.name,
            mu=self.mean,
            sigma=self.stdev,
            lower=lower,
            upper=upper,
            shape=self._get_pymc_shape(),
        )

    def __repr__(self):
        return f"{super().__repr__()} {{mean: {self.mean}, stdev: {self.stdev}, bounds: {self.bounds()}}}"

    @classmethod
    def _get_test(cls):
        return cls("test", 0.0, 1.0, (0.0, 1.0))


class GammaPrior(BasePrior):
    """A gamma distributed prior"""

    def __init__(self, name: str, shape: float, scale: float, size: int = 1):
        super().__init__(name, size)
        self.shape = float(shape)
        self.scale = float(scale)

        self.distri_params = {"shape": self.shape, "scale": self.scale}

        self._rv = stats.gamma(shape, scale=scale)

    def to_pymc(self):
        alpha = self.shape
        beta = 1.0 / self.scale

        return pm.Gamma(self.name, alpha=alpha, beta=beta, shape=self._get_pymc_shape())

    @classmethod
    def from_mode(
        cls,
        name: str,
        mode: float,
        upper_ci: float,
        size: int = 1,
        tol=1e-6,
        max_eval=8,
        warn=False,
    ):
        def evaluate_gamma(params):
            k, theta = params[0], params[1]
            interval = stats.gamma.interval(0.99, k, scale=theta)
            eval_mode = (k - 1.0) * theta
            return np.abs(eval_mode - mode) + np.abs(interval[-1] - upper_ci)

        x = np.array((1.0, 1.0))
        cur_eval = 0
        loss = np.inf
        while (loss > tol) and (cur_eval < max_eval):
            res = minimize(
                evaluate_gamma, x, bounds=[(1e-8, np.inf), (1e-8, np.inf)], method="Nelder-Mead"
            )
            loss = evaluate_gamma(res.x) / upper_ci
            x = res.x
            cur_eval += 1

        if loss > tol:
            if warn:
                raise RuntimeWarning(
                    f"Loss of {loss} exceeds specified tolerance {tol}, parameters may be impossible"
                )

        return cls(name, x[0], x[1], size)

    @classmethod
    def from_mean(
        cls,
        name: str,
        mean: float,
        upper_ci: float,
        size: int = 1,
        tol=1e-6,
        max_eval=8,
        warn=False,
    ):
        def evaluate_gamma(params):
            k, theta = params[0], params[1]
            interval = stats.gamma.interval(0.99, k, scale=theta)
            eval_mean = stats.gamma.mean(k, scale=theta)
            # Force assumption for static typecheckers
            eval_mean = cast(float, eval_mean)
            return np.abs(eval_mean - mean) + np.abs(interval[-1] - upper_ci)

        x = np.array((1.0, 1.0))
        cur_eval = 0
        loss = np.inf
        while (loss > tol) and (cur_eval < max_eval):
            res = minimize(
                evaluate_gamma, x, bounds=[(1e-8, np.inf), (1e-8, np.inf)], method="Nelder-Mead"
            )
            loss = evaluate_gamma(res.x) / upper_ci
            x = res.x
            cur_eval += 1

        if loss > tol:
            if warn:
                raise RuntimeWarning(
                    f"Loss of {loss} exceeds specified tolerance {tol}, parameters may be impossible"
                )

        return cls(name, x[0], x[1], size)

    @classmethod
    def _get_test(cls):
        return cls("test", 1.0, 0.5)
