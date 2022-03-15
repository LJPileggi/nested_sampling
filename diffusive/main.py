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

from .particle import gauss, diffusive_loop
from .utilities.loop_params import loop_par
from .utilities.file_encod import file_encod

def main():
    parser = argparse.ArgumentParser(description='Diffusive nested sampling algorithm implementation.')
    parser.add_argument('--max_level', metavar='max_level', type=int, dest='max_level', help='sets highest level in the run.')
    parser.add_argument('--L_per_level', metavar='L_per_level', type=int, dest='L_per_level', help='sets #points needed to create a new level.')
    parser.add_argument('--dim', metavar='dim', type=int, dest='dim', help='sets dimension of explored space.')
    parser.add_argument('--prior_range', metavar='prior_range', type=float, dest='prior_range', help='sets radius of the explored domain.')
    parser.add_argument('--lam', metavar='lam', type=float, dest='lam', help='sets scale length for level weighting.')
    parser.add_argument('--beta', metavar='beta', type=float, dest='beta', help='sets strength of level enforcement factor.')
    parser.add_argument('--quantile', metavar='quantile', type=float, dest='quantile', help='sets likelihood quantile for level creation.')
    parser.add_argument('--MC_step', metavar='MC_step', type=float, dest='MC_step', help='sets step for MC evolution w.r.t. dimension of the space.\nValues <0.01 recommended.')
    parser.add_argument('--record_step', metavar='record_step', type=int, dest='record_step', help='sets \#steps between each likelihood save.\nValues \~ 100 recommended.')
    parser.add_argument('--seed', metavar='seed', type=int, dest='seed', help='sets seed for simulation.')
    parser.add_argument('--n_runs', metavar='n_runs', type=int, dest='n_runs', help='sets #parallel runs of the simulation.')
    parser.add_argument('--search_L_per_level', metavar='search_L_per_level', type=bool, dest='search_L_per_level', help='run a search over several \#L per level. Must be True or False.')
    parser.add_argument('--search_lam', metavar='search_lam', type=bool, dest='search_lam', help='runs a search over several lambda values. Must be True or False.')
    parser.add_argument('--search_beta', metavar='search_beta', type=bool, dest='search_beta', help='runs a search over several beta values. Must be True or False.')
    parser.add_argument('--search_quantile', metavar='search_quantile', type=bool, dest='search_quantile', help='runs a search over several quantile values. Must be True or False.')
    parser.add_argument('--levels_plot', metavar='levels_plot', type=bool, dest='levels_plot', help='print plot of visited levels.')
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
    parser.set_defaults(n_runs=18)
    parser.set_defaults(search_L_per_level=False)
    parser.set_defaults(search_lam=False)
    parser.set_defaults(search_beta=False)
    parser.set_defaults(search_quantile=False)
    parser.set_defaults(levels_plot=False)
    args = parser.parse_args()


    no_search = (not args.search_L_per_level) & (not args.search_lam) & (not args.search_beta) & (not args.search_quantile)
    loop_params = loop_par(args, gauss)
    with Pool() as pool:
        try:
            final = pool.starmap(diffusive_loop, loop_params)
        except KeyboardInterrupt:
            pool.terminate()
            print('forced termination.')
            exit()
    out = file_encod(args, no_search)
    with open(out, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['max_level', 'L_per_level', 'levels_finished', 'lam', 'beta', 'quantile', 'evidence', 'time taken', 'MC_step'])
        for result in final:
            writer.writerow([result.params.max_level, result.params.L_per_level, result.levels_finished, result.params.lam, result.params.beta, result.params.quantile, result.evidence[-1], result.time, result.params.MC_step])
