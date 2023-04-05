import numpy as np
import pymc as pm
import pytensor.tensor as pt

from estival.model import BayesianCompartmentalModel

def get_wrapped_ll(bcm: BayesianCompartmentalModel):
    """_summary_

    Args:
        bcm: The model to wrap

    Returns:
        A wrapped pytensor op for use in pymc
    """
    # define a pytensor Op for our likelihood function
    class BCMLogLike(pt.Op):
        """
        Specify what type of object will be passed and returned to the Op when it is
        called. In our case we will be passing it a vector of values (the parameters
        that define our model) and returning a single "scalar" value (the
        log-likelihood)
        """

        itypes = [pt.dscalar] * len(bcm.priors)#[pt.dvector]  # expects a vector of parameter values when called
        otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

        def __init__(self):
            """
            Initialise the Op with various things that our log-likelihood function
            requires. Below are the things that are needed in this particular
            example.

            Parameters
            ----------
            bcm: BayesianCompartmentalModel
            """

            # Capture the BayesianCompartmentalModel
            self.bcm = bcm

        def perform(self, node, inputs, outputs):
            params = inputs
            kwargs = {k:params[i] for i,k in enumerate(self.bcm.priors)}

            # call the log-likelihood function
            logl = self.bcm.loglikelihood(**kwargs)

            outputs[0][0] = np.array(logl)  # output the log-likelihood
            
    return BCMLogLike

def use_model(bcm: BayesianCompartmentalModel, include_ll=False) -> list:
    """Use a given BayesianCompartmentalModel for pymc sampling
    This should be called inside a model context like so
    
    with pm.Model():
        variables = use_model(bcm)
        pm.sample(step=[pm.DEMetropolis(variables)])

    Args:
        bcm: The BCM to use for sampling
        include_ll: Include loglikelihood in the sample outputs

    Returns:
        The list of variables to be passed to a sampler step
    """
    logl = get_wrapped_ll(bcm)()

    pymc_priors = []

    for k, prior in bcm.priors.items():
        pymc_priors.append(prior.to_pymc())

    invars = [pt.as_tensor_variable(v) for v in pymc_priors]

    # use a Potential to "call" the Op and include it in the logp computation
    pot = pm.Potential("loglikelihood", logl(*invars))

    if include_ll:
        pm.Deterministic("loglike", pot)
    
    return pymc_priors