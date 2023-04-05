from typing import Tuple, Union
from abc import ABC

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
        return self.rv.interval(ci)

    def ppf(self, q):
        """Probability Percentage Function at q
        Defaults to using the underlying scipy distribution function

        Args:
            q: Quantile (float or arraylike) at which to evaluate ppf

        Returns:
            typeof(q): The ppf values
        """

        return self.rv.ppf(q)

    def logpdf(self, x):
        """Log Probability Density Function at x
        Defaults to using the underlying scipy distribution function

        Args:
            x: Value (float or arraylike) at which to evaluate logpdf

        Returns:
            typeof(x): The logpdf values
        """
        return self.rv.logpdf(x)

    def __repr__(self):
        return f"{type(self)} {self.name}"


# Union type for hierarchical/dispersion parameters
DistriParam = Union[float, BasePrior]


class BetaPrior(BasePrior):
    """
    A beta distributed prior.
    """

    def __init__(self, name: str, mean: float, ci: Tuple[float, float]):
        super().__init__(name)
        self.mean = mean
        self.ci = ci
        self.name = name
        self._find_distri_params()
        self.rv = stats.beta(**self.distri_params)

    def bounds(self, ci=1.0) -> Tuple[float, float]:
        ret = self.rv.interval(ci)
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
        self.rv = stats.uniform(**self.distri_params)
        self.size = size

    def to_pymc(self):
        return pm.Uniform(self.name, lower=self.start, upper=self.end)

class TruncNormalPrior(BasePrior):
    """
    A prior with a truncated normal distribution.
    """

    def __init__(self, name: str, mean: float, stdev: float, trunc_range: Tuple[float, float]):
        super().__init__(name)
        self.mean, self.stdev = mean, stdev
        self.trunc_range = tuple(trunc_range)
        self.distri_params = {
            "loc": self.mean,
            "scale": self.stdev,
            "a": (trunc_range[0] - mean) / stdev,
            "b": (trunc_range[1] - mean) / stdev,
        }
        self.rv = stats.truncnorm(**self.distri_params)

    def to_pymc(self):
        lower, upper = self.trunc_range
        return pm.TruncatedNormal(self.name, mu=self.mean, sigma=self.stdev, lower=lower, upper=upper)