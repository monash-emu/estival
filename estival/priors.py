from typing import Tuple, Union, Dict
from abc import ABC

import numpy as np
from scipy import stats
from scipy.optimize import minimize

# pymc is optional - just be silent on failed import
try:
    import pymc as pm
except:
    pass


class BasePrior(ABC):
    def __init__(self, name: str, size: int = 1):
        self.name = name
        self.size = size

    def bounds(self, ci=1.0) -> Tuple[float, float]:
        return self._rv.interval(ci)

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

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}"

    def to_pymc(self):
        raise NotImplementedError()


# Union type for hierarchical/dispersion parameters
DistriParam = Union[float, BasePrior]

# Dict type used by BayesianCompartmentalModel
PriorDict = Dict[str, BasePrior]


class BetaPrior(BasePrior):
    """
    A beta distributed prior.
    """

    def __init__(self, name: str, mean: float, ci: Tuple[float, float], size=1):
        super().__init__(name, size)
        self.mean = mean
        self.ci = ci
        self.name = name
        self._find_distri_params()
        self._rv = stats.beta(**self.distri_params)

    def bounds(self, ci=1.0) -> Tuple[float, float]:
        ret = self._rv.interval(ci)
        if isinstance(ci, float):
            return ret[0][0], ret[1][0]
        else:
            return ret

    def _find_distri_params(self, ci_width=0.95):
        ci = self.ci
        mean = self.mean

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
        self.distri_params = {"a": best_a, "b": best_b}


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

    def to_pymc(self):
        if self.size > 1:
            lower = np.repeat(self.start, self.size)
            upper = np.repeat(self.end, self.size)
        else:
            lower, upper = self.start, self.end
        return pm.Uniform(self.name, lower=lower, upper=upper)

    def __repr__(self):
        return f"{super().__repr__()} {{bounds: {self.bounds()}}}"


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
            "loc": self.mean,
            "scale": self.stdev,
            "a": (trunc_range[0] - mean) / stdev,
            "b": (trunc_range[1] - mean) / stdev,
        }
        self._rv = stats.truncnorm(**self.distri_params)

    def to_pymc(self):
        lower, upper = self.trunc_range
        if self.size > 1:
            lower = np.repeat(lower, self.size)
            upper = np.repeat(upper, self.size)
        return pm.TruncatedNormal(
            self.name, mu=self.mean, sigma=self.stdev, lower=lower, upper=upper
        )

    def __repr__(self):
        return f"{super().__repr__()} {{mean: {self.mean}, stdev: {self.stdev}, bounds: {self.bounds()}}}"


class GammaPrior(BasePrior):
    """A gamma distributed prior"""

    def __init__(self, name: str, shape: float, scale: float, size: int = 1):
        super().__init__(name, size)
        self.shape = shape
        self.scale = scale

        self._rv = stats.gamma(shape, scale=scale)

    def to_pymc(self):
        alpha = self.shape
        beta = 1.0 / self.scale
        if self.size > 1:
            alpha = np.repeat(alpha, self.size)
            beta = np.repeat(beta, self.size)

        return pm.Gamma(self.name, alpha=alpha, beta=beta)

    @classmethod
    def from_mode(
        cls, name: str, mode: float, upper_ci: float, size: int = 1, tol=1e-6, max_eval=8
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
            raise RuntimeWarning(
                f"Loss of {loss} exceeds specified tolerance {tol}, parameters may be impossible"
            )

        return cls(name, x[0], x[1], size)

    @classmethod
    def from_mean(
        cls, name: str, mean: float, upper_ci: float, size: int = 1, tol=1e-6, max_eval=8
    ):
        def evaluate_gamma(params):
            k, theta = params[0], params[1]
            interval = stats.gamma.interval(0.99, k, scale=theta)
            eval_mean = stats.gamma.mean(k, scale=theta)
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
            raise RuntimeWarning(
                f"Loss of {loss} exceeds specified tolerance {tol}, parameters may be impossible"
            )

        return cls(name, x[0], x[1], size)
