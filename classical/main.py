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

from nested import nested_loop

def main():
    parser = argparse.ArgumentParser(description='Classical Nested Sampling Algorithm.')
    parser.add_argument('--N_iter', metavar='N_iter', type=int, dest='N_iter', help='sets maximum \#iterations.')
    parser.add_argument('--n_points', metavar='n_points', type=int, dest='n_points', help='sets \# points for simulation.')
    parser.add_argument('--dim', metavar='dim', type=int, dest='dim', help='dimension of the explored space.')
    parser.add_argument('--prior_range', metavar='prior_range', type=float, dest='prior_range', help='sets radius of the explored domain.')
    parser.add_argument('--MC_step', metavar='MC_step', type=float, dest='MC_step', help='sets step for MC evolution w.r.t. dimension of the space.\nValues <0.01 recommended.')
    parser.add_argument('--stoch_prior', dest='X_stoch', help='finds new values for prior mass stochastically;\ndefault: takes them according to exp(-i/N).')
    parser.add_argument('--trapezoid', dest='trapezoid', help='sets new weight using trapezoidal rule;\ndefault: takes difference btw consecutive X values.')
    parser.add_argument('--seed', metavar='seed', type=int, dest='seed', help='sets seed for simulation.')
    parser.add_argument('--n_runs', dest='n_runs', type=int, help='\#parallel runs of the simulation;\ndefault: 12.')
    parser.set_defaults(MC_step=0.005)
    parser.set_defaults(X_stoch=False)
    parser.set_defaults(trapezoid=False)
    parser.set_defaults(seed=1)
    parser.set_defaults(n_runs=12)
    args = parser.parse_args()

    params = [(
    args.N_iter,
    args.seed+i,
    args.n_points,
    args.dim,
    args.prior_range,
    args.MC_step,
    args.X_stoch,
    args.trapezoid) for i in range(args.n_runs)]
    with Pool() as pool:
        try:
            results = pool.starmap(nested_loop, params)
        except KeyboardInterrupt:
            pool.terminate()
            print("forced termination")
            exit()
    for result in results:
        print(f'total n. of iterations: {len(result.weights)}; last prior mass value recorded: {result.prior_mass[-1]};')
        print(f'value of evidence obtained: {result.evidence[-1]}; last likelihood value: {result.worst_L}.\n')
