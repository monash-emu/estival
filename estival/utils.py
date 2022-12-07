import arviz as az
import numpy as np


def collect_all_priors(priors, targets):
    out_priors = priors
    for t in targets:
        out_priors = out_priors + t.get_priors()

    return out_priors


def to_arviz(chains, burnin: int):
    from estival.calibration.mcmc.adaptive import AdaptiveChain

    if isinstance(chains, AdaptiveChain):
        chains = [chains]

    c_trace = []
    params = [p.name for p in chains[0].priors]
    for c in chains:

        trace_data = {p.name: [] for p in c.priors}
        last_accept = None
        cur_r = None
        for i, r in enumerate(c.results):
            if r.accept:
                last_accept = r
                cur_r = r
            else:
                cur_r = last_accept

            if i >= burnin:
                if cur_r is not None:
                    for k, v in trace_data.items():
                        v.append(cur_r.parameters[k])

        c_trace.append(trace_data)
    all_params = {}
    for p in params:
        all_params[p] = np.stack([c[p] for c in c_trace])
    return az.from_dict(all_params)
