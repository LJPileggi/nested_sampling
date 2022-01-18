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
    parser.set_defaults(lam=10)
    parser.set_defaults(beta=10)
    parser.set_defaults(quantile=np.e**-1)
    parser.set_defaults(MC_step=0.005)
    parser.set_defaults(record_step=100)
    parser.set_defaults(seed=1)
    parser.set_defaults(n_runs=12)
    args = parser.parse_args()

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
    #params = namedtuple("params", params.keys())(*params.values())

    loop_params = [(
    args.seed+i,
    gauss,
    args.dim,
    args.prior_range,
    params) for i in range(args.n_runs)]
    with Pool() as pool:
        try:
            final = pool.starmap(diffusive_loop, loop_params)
        except KeyboardInterrupt:
            pool.terminate()
            print('forced termination.')
            exit()
    for part in final:
        print(part.evidence[-1])
