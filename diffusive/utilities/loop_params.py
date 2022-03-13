import numpy as np

def loop_par(args, func):
    over_L = (100, 200, 350)#(2000, 3500, 5000, 7500, 10000, 20000, 35000,
        #50000, 75000, 100000, 200000, 350000, 500000, 750000)
    over_lam = [1, 2, 5, 10, 25, 50, 75]
    over_beta = [0, 0.5, 1, 2, 5, 7, 10]
    over_quantile = [0.05, 0.1, 0.2,
        np.e**-1, 0.4, 0.5, 0.6, 0.7]

    no_search = (not args.search_L_per_level) & (not args.search_lam) & (not args.search_beta) & (not args.search_quantile)

    if no_search:
        params = [{
        "max_level" : args.max_level,
        "L_per_level" : args.L_per_level,
        "max_recorded_points" : args.L_per_level*5,
        "C1" : args.L_per_level//10,
        "record_step" : args.record_step,
        "lam" : args.lam,
        "beta" : args.beta,
        "quantile" : args.quantile,
        "MC_step" : args.MC_step}]
    elif args.search_L_per_level:
        params = [
        {"max_level" : args.max_level,
        "L_per_level" : L_per_level,
        "max_recorded_points" : L_per_level*5,
        "C1" : L_per_level//10,
        "record_step" : args.record_step,
        "lam" : args.lam,
        "beta" : args.beta,
        "quantile" : args.quantile,
        "MC_step" : args.MC_step}
        for L_per_level in over_L]
    elif args.search_lam:
        params = [{
        "max_level" : args.max_level,
        "L_per_level" : args.L_per_level,
        "max_recorded_points" : args.L_per_level*5,
        "C1" : args.L_per_level//10,
        "record_step" : args.record_step,
        "lam" : lam,
        "beta" : args.beta,
        "quantile" : args.quantile,
        "MC_step" : args.MC_step} for lam in over_lam]
    elif args.search_beta:
        params = [{
        "max_level" : args.max_level,
        "L_per_level" : args.L_per_level,
        "max_recorded_points" : args.L_per_level*5,
        "C1" : args.L_per_level//10,
        "record_step" : args.record_step,
        "lam" : args.lam,
        "beta" : beta,
        "quantile" : args.quantile,
        "MC_step" : args.MC_step} for beta in over_beta]
    elif args.search_quantile:
        params = [{
        "max_level" : args.max_level,
        "L_per_level" : args.L_per_level,
        "max_recorded_points" : args.L_per_level*5,
        "C1" : args.L_per_level//10,
        "record_step" : args.record_step,
        "lam" : args.lam,
        "beta" : args.beta,
        "quantile" : quantile,
        "MC_step" : args.MC_step} for quantile in over_quantile]
    else:
        raise ValueError('ValueError: parameter not present in model.')
    loop_params = [(
    args.seed+i,
    func,
    args.dim,
    args.prior_range,
    param,
    no_search,
    args.levels_plot) 
    for param in params
    for i in range(args.n_runs)]

    return loop_params
