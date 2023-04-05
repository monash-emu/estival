import arviz as az
import numpy as np
import pandas as pd


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


def _to_df(mcmc, burnin=0, full_trace=False, include_rejected=False):
    out_dict = {p.name: [] for p in mcmc.priors}
    out_dict["log_posterior"] = []
    out_dict["log_likelihood"] = []
    out_dict["iteration"] = []
    out_dict["accepted"] = []

    cur_accepted = mcmc.results[0]

    for i, r in enumerate(mcmc.results):
        add_this = full_trace or include_rejected

        if r.accept:
            cur_accepted = r
            add_this = True

        if include_rejected:
            cur_accepted = r

        if i >= burnin:
            if add_this:
                for k, v in cur_accepted.parameters.items():
                    out_dict[k].append(v)
                out_dict["log_posterior"].append(cur_accepted.lp)
                out_dict["log_likelihood"].append(cur_accepted.ll)
                out_dict["iteration"].append(i)
                out_dict["accepted"].append(r.accept)

    return pd.DataFrame(out_dict)


def to_df(chains, burnin=0, full_trace=False, include_rejected=False):
    from estival.calibration.mcmc.adaptive import AdaptiveChain

    if isinstance(chains, AdaptiveChain):
        return _to_df(chains, burnin, full_trace, include_rejected)
    chain_dfs = []
    for i, chain_res in enumerate(chains):
        chain_df = _to_df(chain_res, burnin, full_trace, include_rejected)
        chain_df["chain"] = i
        chain_dfs.append(chain_df)
    return pd.concat(chain_dfs)

def negative(f, *args, **kwargs):
    """Wrap a positive function such that a minimizable version is returned instead

    Args:
        f: The callable to wrap
    """

    def _reflected(*args, **kwargs):
        return float(0.0 - f(*args, **kwargs))
    
    return _reflected
