from typing import List, Callable, Dict
from time import time
from datetime import timedelta
import logging
from collections import namedtuple

import pandas as pd
import numpy as np
import arviz

from summer2.utils import ref_times_to_dti

from estival.priors import BasePrior
from estival.targets import BaseTarget
from estival.utils import collect_all_priors
from .transformations import TransformedPrior
from .covariance import RunningCovariance

# class AdaptiveMCMC:
#    def __init__(self, priors, targets, model)

DEFAULT_HAARIO_SCALING_FACTOR = 2.4
DEFAULT_METRO_STEP = 0.1

DEFAULT_STEPS = 50

ADAPTIVE_METROPOLIS = {
    "EPSILON": 0.0,
    "MIN_ACCEPTANCE_RATIO": 0.1,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MCMCRes = namedtuple("MCMCRes", "parameters ll lp aq accept run_num")


class _AdaptiveMCMC:
    def __init__(
        self,
        num_chains: int,
        build_model: Callable,
        model_base_parameters: dict,
        priors: List[BasePrior],
        targets: List[BaseTarget],
        initial_priors: dict,
        build_model_kwargs: dict = None,
        chain_id: int = 0,
        haario_scaling_factor: float = DEFAULT_HAARIO_SCALING_FACTOR,
        adaptive_proposal: bool = True,
        metropolis_init_rel_step_size: float = DEFAULT_METRO_STEP,
        fixed_proposal_steps: int = DEFAULT_STEPS,
        seed: int = 0,
        initial_jumping_stdev_ratio: float = 0.25,
        jumping_stdev_adjustment: float = 0.5,
    ):
        pass
        # WIP


class AdaptiveChain:
    """
    Handles model calibration.

    If sampling from the posterior distribution is required, uses a Bayesian algorithm.
    If only one calibrated parameter set is required, uses maximum likelihood estimation.

    A Metropolis Hastings algorithm is used with or without adaptive proposal function.
    The adaptive approach employed was published by Haario et al.

        'An  adaptive  Metropolis  algorithm', Bernoulli 7(2), 2001, 223-242

    """

    def __init__(
        self,
        build_model: Callable,
        model_base_parameters: dict,
        priors: List[BasePrior],
        targets: List[BaseTarget],
        initial_priors: dict,
        build_model_kwargs: dict = None,
        chain_id: int = 0,
        haario_scaling_factor: float = DEFAULT_HAARIO_SCALING_FACTOR,
        adaptive_proposal: bool = True,
        metropolis_init_rel_step_size: float = DEFAULT_METRO_STEP,
        fixed_proposal_steps: int = DEFAULT_STEPS,
        seed: int = 0,
        initial_jumping_stdev_ratio: float = 0.25,
        jumping_stdev_adjustment: float = 0.5,
        initial_covariance=None,
    ):
        """
        Defines a new calibration.
        """
        self.priors = collect_all_priors(priors, targets)

        # self.targets = [t.to_dict() for t in targets]
        # self.targets = remove_early_points_to_prevent_crash(targets, self.all_priors)
        self.targets = targets

        self.haario_scaling_factor = haario_scaling_factor
        self.adaptive_proposal = adaptive_proposal
        self.initial_priors = initial_priors
        self.metropolis_init_rel_step_size = metropolis_init_rel_step_size
        self.n_steps_fixed_proposal = fixed_proposal_steps
        self.initial_jumping_stdev_ratio = initial_jumping_stdev_ratio
        self.jumping_stdev_adjustment = jumping_stdev_adjustment

        self.chain_id = chain_id
        self.seed = seed

        self._prepare_model(build_model, model_base_parameters, build_model_kwargs)

        self._has_run = False

        if initial_covariance is None:
            initial_covariance = np.zeros((len(self.priors), len(self.priors)))
            initial_mean = np.zeros(len(self.priors))
            self.covariance = RunningCovariance(initial_covariance, initial_mean, 1.0)
        else:
            self.covariance = initial_covariance

        self.build_transformations()

    def _prepare_model(self, build_model, model_base_parameters, build_model_kwargs):
        """Prepare for a run, so that all state is valid for calling loglikelihood, logprior etc

        Args:
            project: The Project for this run
            extra_derived (optional): List of derived outputs to generate (additional to targets)
        """

        # Figure out which derived outputs we have to calculate.
        target_names = [t.name for t in self.targets]
        derived_outputs_whitelist = list(set(target_names))

        if build_model_kwargs is None:
            build_model_kwargs = {}

        self.model = build_model(**build_model_kwargs)
        self.model.set_derived_outputs_whitelist(derived_outputs_whitelist)

        model_input_params = self.model.get_input_parameters()

        # Get the keys of the calibration parameters that are directly passed to the model
        # (rather than dispersion/hyperparameters etc)
        model_dyn_params = model_input_params.intersection(set([p.name for p in self.priors]))
        self.calibrated_model_parameters = model_dyn_params

        self.model_runner = self.model.get_runner(model_base_parameters, model_dyn_params)

        # Validate target output start time.
        # self.validate_target_times()

        # Set a custom end time for all model runs - there is no point running
        # the models after the last calibration targets.
        # self.end_time = 2 + max([max(t.data.index) for t in self.targets])

        # work out missing distribution params for priors
        # specify_missing_prior_params(self.iterative_sampling_priors)

        # rebuild self.all_priors, following changes to the two sets of priors
        # self.all_priors = self.iterative_sampling_priors + self.independent_sampling_priors

        # self.update_time_weights()  # for likelihood weighting
        self.workout_unspecified_jumping_stdevs()  # for proposal function definition
        self.param_bounds = {p.name: p.bounds() for p in self.priors}

        # Build target evaluators

        model_times_dti = ref_times_to_dti(self.model.ref_date, self.model.times)

        self.target_evaluators = {t.name: t.get_evaluator(model_times_dti) for t in self.targets}

    def run(
        self,
        available_time: str = None,
        max_iter: int = None,
    ):
        if not self._has_run:
            self.starting_point = self.get_transformed_start(self.initial_priors)

            # Set chain specific seed
            # Chain 0 will have seed equal to that set in the calibration initialisation
            self.seed_chain = ((self.chain_id * 2**22) + self.seed) % (2**32)

            self.mcmc_trace_matrix = None  # will store the results of the MCMC model calibration

            np.random.seed(self.seed_chain)

            # Actually run the calibration
            self.run_autumn_mcmc(available_time, max_iter)
            self._has_run = True
        else:
            logger.info(f"Resuming run for {max_iter} iterations")
            self.enter_mcmc_loop(max_iters=max_iter + self.n_iters_real)

    def run_model_with_params(self, proposed_params: dict):
        """
        Run the model with a set of params.
        """
        # Update default parameters to use calibration params.
        return self.model_runner.run(proposed_params)["derived_outputs"]

    def loglikelihood(self, all_params_dict):
        """
        Calculate the loglikelihood for a set of parameters
        """

        modelled = self.run_model_with_params(all_params_dict)

        ll = 0.0  # loglikelihood if using bayesian approach.

        for k, te in self.target_evaluators.items():
            mdata = modelled[k]
            ll += te.evaluate(mdata, all_params_dict)

        return ll

    def workout_unspecified_time_weights(self):
        """
        Will assign a weight to each time point of each calibration target. If no weights were requested, we will use
        1/n for each time point, where n is the number of time points.
        If a list of weights was specified, it will be rescaled so the weights sum to 1.
        """
        for i, target in enumerate(self.targets):
            if target.time_weights is None:
                target.time_weights = np.ones(len(target.data)) / len(target.data)
            else:
                assert len(target.time_weights) == len(target.data)
                s = sum(target.time_weights)
                target.time_weights = target.time_weights / s

    def workout_unspecified_jumping_stdevs(self):
        self.jumping_sds = {}

        for i, prior in enumerate(self.priors):
            prior_low, prior_high = prior.bounds(0.95)
            prior_width = prior_high - prior_low

            #  95% of the sampled values within [mu - 2*sd, mu + 2*sd], i.e. interval of witdth 4*sd
            relative_prior_width = (
                self.metropolis_init_rel_step_size  # fraction of prior_width in which 95% of samples should fall
            )
            self.jumping_sds[prior.name] = (
                relative_prior_width * prior_width * self.initial_jumping_stdev_ratio
            )

    def validate_in_prior_support(self, iterative_params):
        for i, prior in enumerate(self.priors):
            # Work out bounds for acceptable values, using the support of the prior distribution
            lower_bound, upper_bound = prior.bounds()
            if iterative_params[i] < lower_bound or iterative_params[i] > upper_bound:
                raise Exception(
                    "Sample generated outside prior support", prior, iterative_params[i]
                )

    def run_autumn_mcmc(self, available_time: str = None, max_iters: int = None):
        """
        Run our hand-rolled MCMC algorithm to calibrate model parameters.
        """

        self.mcmc_trace_matrix = None  # will store param trace and loglikelihood evolution

        self.last_accepted_iterative_params_trans = None
        self.last_acceptance_quantity = None  # acceptance quantity is defined as loglike + logprior
        self.n_accepted = 0
        self.n_iters_real = 0  # Actual number of iterations completed, as opposed to run_num.
        self.run_num = 0  # Canonical id of the MCMC run, will be the same as iters until reset by adaptive algo.

        self.results = []

        self.enter_mcmc_loop(available_time, max_iters)

    def enter_mcmc_loop(self, available_time: str = None, max_iters: int = None):
        start_time = time()

        if not available_time and not max_iters:
            raise ValueError("At least one of available_time or max_iters must be specified")

        if available_time:
            available_time = pd.to_timedelta(available_time) / timedelta(seconds=1)

        if max_iters:
            if self.n_iters_real >= max_iters:
                msg = f"Not resuming run. Existing run already has {self.n_iters_real} iterations; max_iters = {max_iters}"
                logger.info(msg)
                return

        while True:
            if (self.run_num % 1000) == 0:
                logger.info(
                    "Chain %s: running iteration %s, run %s",
                    self.chain_id,
                    self.n_iters_real,
                    self.run_num,
                )

            # Propose new parameter set.
            proposed_iterative_params_trans = self.propose_new_iterative_params_trans(
                self.last_accepted_iterative_params_trans, self.haario_scaling_factor
            )
            proposed_iterative_params = self.get_original_params(proposed_iterative_params_trans)

            # Will raise an exception if an invalid sample is generated
            self.validate_in_prior_support(proposed_iterative_params)

            # combine all sampled params into a single dictionary
            iterative_samples_dict = {
                self.priors[i].name: proposed_iterative_params[i] for i in range(len(self.priors))
            }
            all_params_dict = iterative_samples_dict

            # Evaluate log-likelihood.
            proposed_loglike = self.loglikelihood(all_params_dict)

            # Evaluate log-prior.
            proposed_logprior = self.logprior(all_params_dict)

            # posterior distribution
            proposed_log_posterior = proposed_loglike + proposed_logprior

            # transform the density
            proposed_acceptance_quantity = proposed_log_posterior

            for i, prior in enumerate(
                self.priors
            ):  # multiply the density with the determinant of the Jacobian
                inv_derivative = self.transform[prior.name].inverse_derivative(
                    proposed_iterative_params_trans[i]
                )
                if inv_derivative > 0:
                    proposed_acceptance_quantity += np.log(inv_derivative)
                else:
                    proposed_acceptance_quantity += np.log(1.0e-100)

            is_auto_accept = (
                self.last_acceptance_quantity is None
                or proposed_acceptance_quantity >= self.last_acceptance_quantity
            )
            if is_auto_accept:
                accept = True
            else:
                accept_prob = np.exp(proposed_acceptance_quantity - self.last_acceptance_quantity)
                accept = (np.random.binomial(n=1, p=accept_prob, size=1) > 0)[0]

            # Update stored quantities.
            if accept:
                self.last_accepted_iterative_params_trans = proposed_iterative_params_trans
                self.last_acceptance_quantity = proposed_acceptance_quantity
                self.n_accepted += 1

            self.update_mcmc_trace(self.last_accepted_iterative_params_trans)

            # Store model outputs
            self.results.append(
                MCMCRes(
                    all_params_dict,
                    proposed_loglike,
                    proposed_log_posterior,
                    proposed_acceptance_quantity,
                    accept,
                    self.run_num,
                )
            )

            self.run_num += 1
            self.n_iters_real += 1
            if available_time:
                # Stop iterating if we have run out of time.
                elapsed_time = time() - start_time
                if elapsed_time > available_time:
                    msg = f"Stopping MCMC simulation after {self.n_iters_real} iterations because of {available_time}s time limit"
                    logger.info(msg)
                    break
            if max_iters:
                # Stop running if we have performed enough iterations
                if self.n_iters_real >= max_iters:
                    msg = f"Stopping MCMC simulation after {self.n_iters_real} iterations, maximum iterations hit"
                    logger.info(msg)
                    break

            # Check that the pre-adaptive phase ended with a decent acceptance ratio
            if self.n_steps_fixed_proposal > 0:
                if self.adaptive_proposal and self.run_num == self.n_steps_fixed_proposal:
                    acceptance_ratio = self.n_accepted / self.run_num
                    logger.info(
                        "Pre-adaptive phase completed at %s iterations after %s runs with an acceptance ratio of %s.",
                        self.n_iters_real,
                        self.run_num,
                        acceptance_ratio,
                    )
                    if acceptance_ratio < ADAPTIVE_METROPOLIS["MIN_ACCEPTANCE_RATIO"]:
                        logger.info("Acceptance ratio too low, restart sampling from scratch.")
                        (
                            self.run_num,
                            self.n_accepted,
                            self.last_accepted_params_trans,
                            self.last_acceptance_quantity,
                        ) = (0, 0, None, None)
                        self.reduce_proposal_step_size()
                    else:
                        logger.info("Acceptance ratio acceptable, continue sampling.")

    def reduce_proposal_step_size(self):
        """
        Reduce the "jumping_stdev" associated with each parameter during the pre-adaptive phase
        """
        self.jumping_sds = {
            k: v * self.jumping_stdev_adjustment for k, v in self.jumping_sds.items()
        }

    def build_adaptive_covariance_matrix(self, haario_scaling_factor):
        scaling_factor = haario_scaling_factor**2 / len(self.priors)  # from Haario et al. 2001
        # cov_matrix = np.cov(self.mcmc_trace_matrix, rowvar=False)
        cov_matrix = self.covariance.cov_t
        adaptive_cov_matrix = scaling_factor * cov_matrix + scaling_factor * ADAPTIVE_METROPOLIS[
            "EPSILON"
        ] * np.eye(len(self.priors))
        return adaptive_cov_matrix

    def build_transformations(self):
        self.transform: Dict[str, TransformedPrior] = {}

        for prior in self.priors:
            param_name = prior.name
            self.transform[param_name] = TransformedPrior(prior)

    def get_transformed_start(self, starting_point: dict):
        """
        Build transformation functions between the parameter space and R^n.
        Additionally, perform any updates to initial point bounding and jumping sds
        """

        starting_point = starting_point.copy()

        for i, prior in enumerate(self.priors):
            param_name = prior.name

            # lower_bound = self.param_bounds[param_name][0]
            # upper_bound = self.param_bounds[param_name][1]
            lower_bound, upper_bound = prior.bounds()

            # we will need to transform the jumping step
            original_sd = self.jumping_sds[param_name]

            # trivial case of an unbounded parameter
            if lower_bound == -float("inf") and upper_bound == float("inf"):
                representative_point = None
            # case of a lower-bounded parameter with infinite support
            elif upper_bound == float("inf"):
                representative_point = lower_bound + 10 * original_sd
                if starting_point[param_name] <= lower_bound:
                    starting_point[param_name] = lower_bound + original_sd / 10

            # case of an upper-bounded parameter with infinite support
            elif lower_bound == -float("inf"):
                representative_point = upper_bound - 10 * original_sd
                if starting_point[param_name] >= upper_bound:
                    starting_point[param_name] = upper_bound - original_sd / 10
            # case of a lower- and upper-bounded parameter
            else:
                representative_point = 0.5 * (lower_bound + upper_bound)
                if starting_point[param_name] <= lower_bound:
                    starting_point[param_name] = lower_bound + original_sd / 10
                elif starting_point[param_name] >= upper_bound:
                    starting_point[param_name] = upper_bound - original_sd / 10

            # Don't update jumping if we are resuming (this has already been calculated)
            # FIXME:  We should probably refactor this to update on copies rather than in place
            if representative_point is not None:
                transformed_low = self.transform[param_name].direct(
                    representative_point - original_sd / 4
                )
                transformed_up = self.transform[param_name].direct(
                    representative_point + original_sd / 4
                )
                self.jumping_sds[param_name] = abs(transformed_up - transformed_low)
        return starting_point

    def get_original_params(self, transformed_iterative_params):
        original_iterative_params = []
        for i, prior in enumerate(self.priors):
            original_iterative_params.append(
                self.transform[prior.name].inverse(transformed_iterative_params[i])
            )
        return original_iterative_params

    def propose_new_iterative_params_trans(
        self, prev_iterative_params_trans, haario_scaling_factor=2.4
    ):
        """
        calculated the joint log prior
        :param prev_iterative_params_trans: last accepted parameter values as a list ordered using the order of
         self.iterative_sampling_priors
        :return: a new list of parameter values
        """
        new_iterative_params_trans = []
        # if this is the initial step
        if prev_iterative_params_trans is None:
            for prior in self.priors:
                start_point = self.starting_point[prior.name]
                new_iterative_params_trans.append(self.transform[prior.name].direct(start_point))
            return new_iterative_params_trans

        use_adaptive_proposal = (
            self.adaptive_proposal and self.run_num > self.n_steps_fixed_proposal
        )

        if use_adaptive_proposal:
            adaptive_cov_matrix = self.build_adaptive_covariance_matrix(haario_scaling_factor)
            if np.all((adaptive_cov_matrix == 0)):
                use_adaptive_proposal = (
                    False  # we can't use the adaptive method for this step as the covariance is 0.
                )
            else:
                new_iterative_params_trans = sample_from_adaptive_gaussian(
                    prev_iterative_params_trans, adaptive_cov_matrix
                )

        if not use_adaptive_proposal:
            for i, prior in enumerate(self.priors):
                sample = np.random.normal(
                    loc=prev_iterative_params_trans[i], scale=self.jumping_sds[prior.name], size=1
                )[0]
                new_iterative_params_trans.append(sample)

        return new_iterative_params_trans

    def logprior(self, all_params_dict):
        """
        calculated the joint log prior
        :param all_params_dict: model parameters as a dictionary
        :return: the natural log of the joint prior
        """
        logp = 0.0
        for p in self.priors:
            value = all_params_dict[p.name]
            logp += p.logpdf(value)

        return logp

    def update_mcmc_trace(self, params_to_store):
        """
        store mcmc iteration into param_trace
        :param params_to_store: model parameters as a list of values ordered using the order of self.iterative_sampling_priors
        :param loglike_to_store: current loglikelihood value
        """
        if self.mcmc_trace_matrix is None:
            self.mcmc_trace_matrix = np.array([params_to_store])
        else:
            self.mcmc_trace_matrix = np.concatenate(
                (self.mcmc_trace_matrix, np.array([params_to_store]))
            )

        self.covariance.update(np.array(params_to_store))

    def to_arviz(self, burnin: int):
        trace_data = {p.name: [] for p in self.priors}
        last_accept = None
        for r in self.results[burnin:]:
            if r.accept:
                last_accept = r
                cur_r = r
            else:
                cur_r = last_accept

            if cur_r is not None:
                for k, v in trace_data.items():
                    v.append(cur_r.parameters[k])
        return arviz.from_dict(trace_data)


def sample_from_adaptive_gaussian(prev_params, adaptive_cov_matrix):
    return np.random.multivariate_normal(prev_params, adaptive_cov_matrix)
