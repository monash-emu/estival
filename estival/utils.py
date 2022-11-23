def collect_all_priors(priors, targets):
    out_priors = priors
    for t in targets:
        out_priors = out_priors + t.get_priors()

    return out_priors
