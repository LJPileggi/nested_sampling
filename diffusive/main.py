import os
import sys
import csv
import random
import math
from math import floor
import argparse

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

from particle import gauss, diffusive_loop

def main():
    parser = argparse.ArgumentParser(description='Diffusive nested sampling algorithm implementation.')
    parser.add_argument('--max_level', metavar='max_level', type=int, dest='max_level', help='sets highest level in the run.')
    parser.add_argument('--L_per_level', metavar='L_per_level', type=int, dest='L_per_level', help='sets \#/points needed to create a new level.')
    parser.add_argument('--dim', metavar='dim', type=int, dest='dim', help='sets dimension of explored space.')
    parser.add_argument('--prior_range', metavar='prior_range', type=float, dest='prior_range', help='sets radius of the explored domain.')
    parser.add_argument('--lam', metavar='lam', type=float, dest='lam', help='sets scale length for level weighting.')
    parser.add_argument('--beta', metavar='beta', type=float, dest='beta', help='sets strength of level enforcement factor.')
    parser.add_argument('--quantile', metavar='quantile', type=float, dest='quantile', help='sets likelihood quantile for level creation.')
    parser.add_argument('--MC_step', metavar='MC_step', type=float, dest='MC_step', help='sets step for MC evolution w.r.t. dimension of the space.\nValues <0.01 recommended.')
    parser.add_argument('--record_step', metavar='record_step', type=int, dest='record_step', help='sets \#steps between each likelihood save.\nValues \~ 100 recommended.')
    parser.add_argument('--seed', metavar='seed', type=int, dest='seed', help='sets seed for simulation.')
    parser.add_argument('--n_runs', metavar='n_runs', type=int, dest='n_runs', help='sets \#parallel runs of the simulation.')
    parser.add_argument('--search_L_per_level', metavar='search_L_per_level', type=bool, dest='search_L_per_level', help='runs a search over several \#L per level.')
    parser.add_argument('--search_lam', metavar='search_lam', type=bool, dest='search_lam', help='runs a search over several lambda values.')
    parser.add_argument('--search_beta', metavar='search_beta', type=bool, dest='search_beta', help='runs a search over several beta values.')
    parser.add_argument('--search_quantile', metavar='search_quantile', type=bool, dest='search_quantile', help='runs a search over several quantile values.')
    parser.set_defaults(max_level=100)
    parser.set_defaults(L_per_level=2000)
    parser.set_defaults(dim=50)
    parser.set_defaults(prior_range=30.)
    parser.set_defaults(lam=10)
    parser.set_defaults(beta=10)
    parser.set_defaults(quantile=np.e**-1)
    parser.set_defaults(MC_step=0.005)
    parser.set_defaults(record_step=100)
    parser.set_defaults(seed=1)
    parser.set_defaults(n_runs=12)
    parser.set_defaults(search_L_per_level=False)
    parser.set_defaults(search_lam=False)
    parser.set_defaults(search_beta=False)
    parser.set_defaults(search_quantile=False)
    args = parser.parse_args()

    over_L = (50, 75, 100, 200, 350, 500, 750,
    1000, 2000, 3500, 5000, 7500, 10000, 20000)

    over_lam = [1, 2, 5, 10, 25, 50, 75]

    over_beta = [0, 0.5, 1, 2, 5, 7, 10]

    over_quantile = [0.05, 0.1, 0.2,
    np.e**-1, 0.4, 0.5, 0.6, 0.7]

    no_search = (not args.search_L_per_level) & (not args.search_lam) & (not args.search_beta) & (not args.search_quantile)

    if no_search:
        params = {
        "max_level" : args.max_level,
        "L_per_level" : args.L_per_level,
        "max_recorded_points" : args.L_per_level*5,
        "C1" : args.L_per_level//10,
        "record_step" : args.record_step,
        "lam" : args.lam,
        "beta" : args.beta,
        "quantile" : args.quantile,
        "MC_step" : args.MC_step}

        loop_params = [(
        args.seed+i,
        gauss,
        args.dim,
        args.prior_range,
        params,
        no_search) for i in range(args.n_runs)]

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

        loop_params = [(
        args.seed+i,
        gauss,
        args.dim,
        args.prior_range,
        param,
        no_search) 
        for param in params
        for i in range(args.n_runs)]

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

        loop_params = [(
        args.seed+i,
        gauss,
        args.dim,
        args.prior_range,
        param,
        no_search) 
        for param in params
        for i in range(args.n_runs)]

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

        loop_params = [(
        args.seed+i,
        gauss,
        args.dim,
        args.prior_range,
        param,
        no_search) 
        for param in params
        for i in range(args.n_runs)]

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

        loop_params = [(
        args.seed+i,
        gauss,
        args.dim,
        args.prior_range,
        param,
        no_search) 
        for param in params
        for i in range(args.n_runs)]

    with Pool() as pool:
        try:
            final = pool.starmap(diffusive_loop, loop_params)
        except KeyboardInterrupt:
            pool.terminate()
            print('forced termination.')
            exit()
