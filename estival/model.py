from typing import List
from dataclasses import dataclass

from summer2 import CompartmentalModel

from jax import jit

import pandas as pd


class BayesianCompartmentalModel:
    def __init__(
        self,
        model: CompartmentalModel,
        parameters: dict,
        priors: list,
        targets: list,
        extra_ll=None,
    ):
        self.model = model
        self.parameters = parameters
        self.targets = {t.name: t for t in targets}

        for t in targets:
            priors = priors + t.get_priors()
        self.priors = {p.name: p for p in priors}

        self.loglikelihood = self._build_logll_func(extra_ll)

        self._ref_idx = self.model._get_ref_idx()

    def _build_logll_func(self, extra_ll=None):
        model_params = self.model.get_input_parameters()
        dyn_params = list(model_params.intersection(set(self.priors)))
        self.model.set_derived_outputs_whitelist(list(self.targets))

        self._ll_runner = self.model.get_runner(self.parameters, dyn_params)

        self.model.set_derived_outputs_whitelist([])
        self._full_runner = self.model.get_runner(self.parameters, dyn_params)

        tidx = pd.Index(self.model.times)

        self._evaluators = {}
        for k, t in self.targets.items():
            tev = t.get_evaluator(tidx)
            self._evaluators[k] = tev.evaluate

        @jit
        def logll(**kwargs):
            dict_args = capture_model_kwargs(self.model, **kwargs)
            res = self._ll_runner._run_func(dict_args)["derived_outputs"]

            logdens = 0.0
            for tk, te in self._evaluators.items():
                modelled = res[tk]
                logdens += te(modelled, kwargs)

            if extra_ll:
                logdens += extra_ll(kwargs)

            return logdens

        return logll

    def logprior(self, **parameters):
        lp = 0.0
        for k, p in self.priors.items():
            lp += p.logpdf(parameters[k])
        return lp

    def logposterior(self, **parameters):
        return self.loglikelihood(**parameters) + self.logprior(**parameters)

    def run(self, parameters):
        results = self._full_runner._run_func(parameters)
        return ResultsData(
            derived_outputs=pd.DataFrame(results["derived_outputs"], index=self._ref_idx)
        )

    def run_jax(self, parameters):
        return self._full_runner._run_func(parameters)


@dataclass
class ResultsData:
    derived_outputs: pd.DataFrame


def capture_model_kwargs(model: CompartmentalModel, **kwargs) -> dict:
    model_params = model.get_input_parameters()
    return {k: kwargs[k] for k in kwargs if k in model_params}
