from typing import List, Dict
from dataclasses import dataclass

from summer2 import CompartmentalModel

from jax import jit
import numpy as np

import pandas as pd

from .targets import BaseTarget
from .priors import BasePrior


@dataclass
class ResultsData:
    derived_outputs: pd.DataFrame
    extras: dict


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

        self._model_parameters = model.get_input_parameters()

        self.parameters = parameters
        self.targets: Dict[str, BaseTarget] = _named_list_to_dict(targets)

        for t in targets:
            priors = priors + t.get_priors()
        self.priors: Dict[str, BasePrior] = _named_list_to_dict(priors)

        self._ref_idx = self.model._get_ref_idx()
        if not isinstance(self._ref_idx, pd.Index):
            self._ref_idx = pd.Index(self._ref_idx)
        self.epoch = self.model.get_epoch()

        self._build_logll_funcs(extra_ll)

        from .utils.sample import SampledPriorsManager

        self.sample = SampledPriorsManager(self.priors)

    def _construct_targets(self, targets: list) -> Dict[str, BaseTarget]:
        tdict = {}

        for t in targets:
            if t.name in tdict:
                raise ValueError("Duplicate target name", t.name)
            else:
                tdict[t.name] = t

        return tdict

    def _build_logll_funcs(self, extra_ll=None):
        model_params = self.model.get_input_parameters()
        dyn_params = list(model_params.intersection(set(self.priors)))
        self.model.set_derived_outputs_whitelist(
            list(set([t.model_key for t in self.targets.values()]))
        )

        self._ll_runner = self.model.get_runner(
            self.parameters, dyn_params, include_full_outputs=False
        )

        self.model.set_derived_outputs_whitelist([])
        self._full_runner = self.model.get_runner(
            self.parameters, dyn_params, include_full_outputs=False
        )

        self._evaluators = {}
        for k, t in self.targets.items():
            tev = t.get_evaluator(self._ref_idx, self.epoch)
            self._evaluators[k] = tev.evaluate

        @jit
        def logll(**kwargs):
            dict_args = capture_model_kwargs(self.model, **kwargs)
            res = self._ll_runner._run_func(dict_args)["derived_outputs"]

            logdens = 0.0
            for tname, target in self.targets.items():
                model_key = target.model_key
                evaluator = self._evaluators[tname]
                modelled = res[model_key]
                logdens += evaluator(modelled, kwargs)

            if extra_ll:
                logdens += extra_ll(kwargs)

            return logdens

        logll.__doc__ = f"""logll({', '.join([k for k in self.priors])})\n
        Run the model for a given set of parameters, and 
        return the loglikelihood of its outputs, including any values from extrall"""

        @jit
        def logll_multi(modelled_do, **kwargs):
            out_ll = {}

            for tname, target in self.targets.items():
                model_key = target.model_key
                evaluator = self._evaluators[tname]
                modelled = modelled_do[model_key]
                out_ll[tname] = evaluator(modelled, kwargs)

            if extra_ll:
                out_ll["extra_ll"] = extra_ll(kwargs)

            return out_ll

        self._logll_multi = logll_multi
        self.loglikelihood = logll

    def logprior(self, **parameters):
        lp = 0.0
        for k, p in self.priors.items():
            lp += np.sum(p.logpdf(parameters[k]))
        return lp

    def logposterior(self, **parameters):
        return self.loglikelihood(**parameters) + self.logprior(**parameters)

    def run(self, parameters: dict, include_extras=True) -> ResultsData:
        """Run the model for a given set of parameters.
        Note that only parameters specified as priors affect the outputs; other parameters
        are in-filled from the initial arguments supplied to BayesianCompartmentalModel

        Args:
            parameters: Dict of parameter key/values (as specified in priors)

        Returns:
            ResultsData, an extensible container with derived_outputs as a DataFrame
        """
        run_params = {k: v for k, v in parameters.items() if k in self._model_parameters}
        results = self._full_runner._run_func(run_params)

        if include_extras:
            extras = {}
            ll_components = self._logll_multi(results["derived_outputs"], **parameters)
            extras["ll_components"] = ll_components
            extras["loglikelihood"] = sum(ll_components.values())
            extras["logprior"] = self.logprior(**parameters)
            extras["logposterior"] = extras["logprior"] + extras["loglikelihood"]
        else:
            extras = {}

        return ResultsData(
            derived_outputs=pd.DataFrame(results["derived_outputs"], index=self._ref_idx),
            extras=extras,
        )

    def run_jax(self, parameters: dict) -> dict:
        """Run the jax run function for the model directly with the supplied parameters;
        meaning bcm.run_jax can be included in JIT calls

        Args:
            parameters: Dict of parameter key/values (as specified in priors)

        Returns:
            Results as per the summer2 jax runner
        """
        return self._full_runner._run_func(parameters)


def capture_model_kwargs(model: CompartmentalModel, **kwargs) -> dict:
    model_params = model.get_input_parameters()
    return {k: kwargs[k] for k in kwargs if k in model_params}


def _named_list_to_dict(in_list: list) -> dict:
    tdict = {}

    for t in in_list:
        if t.name in tdict:
            raise ValueError("Duplicate name in input list", t.name)
        else:
            tdict[t.name] = t

    return tdict
