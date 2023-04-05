from numpy import exp, log
from jax.scipy.special import logit, expit


def make_transform_func_with_lower_bound(lower_bound, func_type):
    assert func_type in ["direct", "inverse", "inverse_derivative"]
    if func_type == "direct":

        def func(x):
            return log(x - lower_bound)

    elif func_type == "inverse":

        def func(y):
            return exp(y) + lower_bound

    elif func_type == "inverse_derivative":

        def func(y):
            return exp(y)

    return func


def make_transform_func_with_upper_bound(upper_bound, func_type):
    assert func_type in ["direct", "inverse", "inverse_derivative"]
    if func_type == "direct":

        def func(x):
            return log(upper_bound - x)

    elif func_type == "inverse":

        def func(y):
            return upper_bound - exp(y)

    elif func_type == "inverse_derivative":

        def func(y):
            return exp(y)

    return func


def make_transform_func_with_two_bounds(lower_bound, upper_bound, func_type):
    assert func_type in ["direct", "inverse", "inverse_derivative"]
    if func_type == "direct":

        def func(x):
            return logit((x - lower_bound) / (upper_bound - lower_bound))

    elif func_type == "inverse":

        def func(y):
            # prevent overflow when the parameter gets close to its bounds (i.e. transformed param's norm gets too big)
            return lower_bound + (upper_bound - lower_bound) * expit(y)

    elif func_type == "inverse_derivative":

        def func(y):
            inv_logit = expit(y)
            return (upper_bound - lower_bound) * inv_logit * (1.0 - inv_logit)

    return func


class TransformedPrior:
    def __init__(self, original_prior):
        self.prior = original_prior

        transformations = build_transformations(original_prior)
        self.direct = transformations["direct"]
        self.inverse = transformations["inverse"]
        self.inverse_derivative = transformations["inverse_derivative"]


def build_transformations(prior):
    """
    Build transformation functions between the parameter space and R^n.
    """
    transform = {
        "direct": None,  # param support to R
        "inverse": None,  # R to param space
        "inverse_derivative": None,  # R to R
    }
    lower_bound, upper_bound = prior.bounds()

    # trivial case of an unbounded parameter
    if lower_bound == -float("inf") and upper_bound == float("inf"):
        transform["direct"] = lambda x: x
        transform["inverse"] = lambda x: x
        transform["inverse_derivative"] = lambda x: 1.0

    # case of a lower-bounded parameter with infinite support
    elif upper_bound == float("inf"):
        for func_type in ["direct", "inverse", "inverse_derivative"]:
            transform[func_type] = make_transform_func_with_lower_bound(lower_bound, func_type)

    # case of an upper-bounded parameter with infinite support
    elif lower_bound == -float("inf"):
        for func_type in ["direct", "inverse", "inverse_derivative"]:
            transform[func_type] = make_transform_func_with_upper_bound(upper_bound, func_type)

    # case of a lower- and upper-bounded parameter
    else:
        for func_type in ["direct", "inverse", "inverse_derivative"]:
            transform[func_type] = make_transform_func_with_two_bounds(
                lower_bound, upper_bound, func_type
            )

    return transform
