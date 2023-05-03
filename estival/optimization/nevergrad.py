from typing import Callable

from concurrent import futures
from multiprocessing import cpu_count
import numpy as np

from estival.model import BayesianCompartmentalModel
from estival.utils import negative


# This is optional, just be silent if it's not installed...
try:
    import nevergrad as ng
except:
    pass


def get_instrumentation(priors, suggested=None, init_method="midpoint"):
    idict = {}

    suggested = suggested or {}
    starting_points = {}
    for pk, p in priors.items():
        if sug_pt := suggested.get(pk):
            starting_points[pk] = sug_pt
        else:
            if init_method == "midpoint":
                pp = np.repeat(0.5, p.size)
            elif init_method == "uniform":
                pp = np.random.uniform(size=p.size)
            else:
                raise ValueError("Invalid init method", init_method)
            starting_points[pk] = p.ppf(pp)

    for pk, p in priors.items():
        lower, upper = p.bounds()
        if p.size == 1:
            idict[pk] = ng.p.Scalar(init=starting_points[pk], lower=lower, upper=upper)
        else:
            idict[pk] = ng.p.Array(init=starting_points[pk], lower=lower, upper=upper)

    return ng.p.Instrumentation(**idict)


class OptRunner:
    def __init__(self, optimizer, min_func, num_workers):
        self.optimizer = optimizer
        self.min_func = min_func
        self.num_workers = num_workers

    def minimize(self, budget):
        cur_ask = self.optimizer.num_ask
        self.optimizer.budget = cur_ask + budget
        with futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            rec = self.optimizer.minimize(self.min_func, executor=executor)
        return rec


def optimize_model(
    bcm: BayesianCompartmentalModel,
    budget: int = 1000,
    opt_class=ng.optimizers.NGOpt,
    num_workers: int = None,
    suggested: dict = None,
    init_method: str = "midpoint",
    obj_function: Callable = None,
    invert_function=True,
):
    if not num_workers:
        num_workers = cpu_count()

    instrum = get_instrumentation(bcm.priors, suggested, init_method)

    def as_float(wrapped):
        def float_wrapper(**parameters):
            return float(wrapped(**parameters))

        return float_wrapper

    if obj_function is None:
        obj_function = bcm.loglikelihood
    if invert_function:
        obj_function = negative(obj_function)
    else:
        obj_function = as_float(obj_function)

    min_func = obj_function
    optimizer = opt_class(parametrization=instrum, budget=budget, num_workers=num_workers)
    return OptRunner(optimizer, min_func, num_workers)
