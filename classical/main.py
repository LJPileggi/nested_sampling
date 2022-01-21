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
    parser.add_argument('--stoch_prior', dest='X_stoch', type=bool, help='finds new values for prior mass stochastically;\ndefault: takes them according to exp(-i/N).')
    parser.add_argument('--trapezoid', dest='trapezoid', type=bool, help='sets new weight using trapezoidal rule;\ndefault: takes difference btw consecutive X values.')
    parser.add_argument('--seed', metavar='seed', type=int, dest='seed', help='sets seed for simulation.')
    parser.add_argument('--n_runs', dest='n_runs', type=int, help='\#parallel runs of the simulation;\ndefault: 12.')
    parser.add_argument('--automatised', metavar='automatised', type=bool, dest='automatised', help='runs automatically the algorithm over several configurations of \#points.')
    parser.set_defaults(N_iter=20000)
    parser.set_defaults(n_points=100)
    parser.set_defaults(dim=50)
    parser.set_defaults(prior_range=30.)
    parser.set_defaults(MC_step=0.005)
    parser.set_defaults(X_stoch=False)
    parser.set_defaults(trapezoid=False)
    parser.set_defaults(seed=1)
    parser.set_defaults(n_runs=18)
    parser.set_defaults(automatised=False)
    args = parser.parse_args()

    if not args.automatised:
        params = [(
        args.N_iter,
        args.seed+i,
        args.n_points,
        args.dim,
        args.prior_range,
        args.MC_step,
        args.X_stoch,
        args.trapezoid,
        args.automatised) for i in range(args.n_runs)]
    else:
        N_points = [5, 10, 20, 35, 50, 75, 100]#,
        #200, 350, 500, 750, 1000, 2000, 3500]
        N_iter = list(1000*np.array(N_points))
        params = [(
        n_iter,
        args.seed+i,
        n_points,
        args.dim,
        args.prior_range,
        args.MC_step,
        args.X_stoch,
        args.trapezoid,
        args.automatised)
        for n_iter, n_points in zip(N_iter, N_points)
        for i in range(args.n_runs)]
    with Pool() as pool:
        try:
            results = pool.starmap(nested_loop, params)
        except KeyboardInterrupt:
            pool.terminate()
            print("forced termination")
            exit()
    output_path = os.path.abspath('output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if args.automatised & args.X_stoch:
        out = os.path.join(output_path, f'results_loop_X_stoch.csv')
    elif args.automatised & args.trapezoid:
        out = os.path.join(output_path, f'results_loop_trapezoid.csv')
    elif args.automatised:
        out = os.path.join(output_path, f'results_loop.csv')
    elif args.X_stoch:
        out = os.path.join(output_path, f'results_X_stoch.csv')
    elif args.trapezoid:
        out = os.path.join(output_path, f'results_trapezoid.csv')
    else:
        out = os.path.join(output_path, f'results.csv')
    with open(out, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for result in results:
            writer.writerow([result.n_points, result.N_iter, result.evidence[-1]])
