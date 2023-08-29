import pytest

import pymc as pm

from estival import priors as esp

PRIORS = ["BetaPrior", "UniformPrior", "TruncNormalPrior", "GammaPrior"]


def get_test_prior(prior_type: str) -> esp.BasePrior:
    pclass: esp.BasePrior = getattr(esp, prior_type)
    return pclass._get_test()


@pytest.mark.parametrize("prior_type", PRIORS)
def test_to_pymc(prior_type: str):
    p = get_test_prior(prior_type)

    with pm.Model() as m:  # type: ignore
        pm_prior = p.to_pymc()


@pytest.mark.parametrize("prior_type", PRIORS)
@pytest.mark.parametrize("fit_func", ["pdf", "cdf"])
def test_get_series(prior_type: str, fit_func: str):
    p = get_test_prior(prior_type)

    p.get_series(fit_func)
