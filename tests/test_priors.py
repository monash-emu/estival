import pytest


from estival import priors as esp

PRIORS = ["BetaPrior", "UniformPrior", "NormalPrior", "TruncNormalPrior", "GammaPrior"]


def get_test_prior(prior_type: str) -> esp.BasePrior:
    pclass: esp.BasePrior = getattr(esp, prior_type)
    return pclass._get_test()


try:
    import pymc as pm

    @pytest.mark.parametrize("prior_type", PRIORS)
    def test_to_pymc(prior_type: str):
        p = get_test_prior(prior_type)

        with pm.Model() as m:  # type: ignore
            pm_prior = p.to_pymc()

except:
    pass


@pytest.mark.parametrize("prior_type", PRIORS)
@pytest.mark.parametrize("fit_func", ["pdf", "cdf"])
def test_get_series(prior_type: str, fit_func: str):
    p = get_test_prior(prior_type)

    p.get_series(fit_func)
