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

def multiply(*args):
    return np.prod(args)

def polar_nd_init(dim, radius):
    U = np.random.uniform(0., 1.)
    R = radius*U**(1./dim)
    phis = np.random.uniform(0., np.pi, dim-2)
    theta = np.random.uniform(0., 2*np.pi)
    coord = [np.cos(phis[0])]
    i = 1
    while i<dim-2:
        coord.append(np.cos(phis[i]))
        coord[-1] *= multiply(np.sin(phis[0:i]))
        i += 1
    coord.append(np.cos(theta))
    coord[-1] *= multiply(np.sin(phis))
    coord.append(np.sin(theta))
    coord[-1] *= multiply(np.sin(phis))
    coord = np.array(coord)*R
    return coord

def proposal(x_t, len_step, dim):
    x_t1 = x_t + len_step*np.random.uniform(-1., 1., dim)
    return x_t1

def likelihood(x):
    L = np.exp(-(x*x).sum()/2.)
    return L

def MCMC(theta, L_s, dim, len_step, n=100):
    theta_new = theta
    i = 0
    while (i <= n) | (likelihood(theta_new) < L_s):
        theta_prop = proposal(theta_new, len_step, dim)
        if likelihood(theta_prop) > L_s:
            theta_new = theta_prop
            i += 1
    return theta_new

def gauss(point):
    r2 = (point*point).sum()
    return np.exp(-r2/2.)

class point():
    def __init__(self, theta, likelihood):
        self.theta = theta
        self.likelihood = likelihood

class nested():
    def __init__(self, n_points, dim, prior_range, MC_step=0.005, X_stoch=False, trapezoid=False):
        self._n_points = n_points
        self._dim = dim
        self._prior_range = prior_range
        self._MC_step = MC_step
        self._X_stoch = X_stoch
        self._trapezoid = trapezoid
        self.points = []
        i = 0
        while i<self._n_points:
            theta = polar_nd_init(dim, prior_range)
            self.points.append(point(theta, gauss(theta)))
            i += 1
        self.evidence = [0.]
        if not trapezoid:
            self.prior_mass = [1.]
        else:
            self.prior_mass[2.-np.exp(-1./n_points), np.exp(-1./n_points)]
        self.weights = []
        self.worst_L_series = [0.]
        self.worst_idx = 0
        self.worst_L = 0.

    def find_worst(self):
        worst, L_w = 0, self.points[0].likelihood
        i = 1
        while i<self._n_points:
            if self.points[i].likelihood < L_w:
                worst, L_w = i, self.points[i].likelihood
            i += 1
        self.worst_idx, self.worst_L = worst, L_w

    def update_quantities(self, iter_step):
        if not self._X_stoch:
            self.prior_mass.append(np.exp(-iter_step/self._n_points))
        else:
            self.prior_mass.append(self.prior_mass[-1]*np.random.uniform(0., 1.)**(1./(self._n_points-1)))
        if not self._trapezoid:
            self.weights.append(self.prior_mass[iter_step-1]-self.prior_mass[iter_step])
        else:
            self.weights.append((self.prior_mass[iter_step-1]-self.prior_mass[iter_step+1])/2.)
        self.evidence.append(self.evidence[-1] + self.worst_L*self.weights[-1])

    def substitute_worst(self):
        idx_new = random.choice(range(self._n_points))
        while idx_new == self.worst_idx:
            idx_new = random.choice(range(self._n_points))
        theta_new = MCMC(self.points[idx_new].theta, self.worst_L, self._dim, np.sqrt(-2.*np.log(self.worst_L))*self._MC_step)
        self.points[self.worst_idx] = point(theta_new, gauss(theta_new))

    def final_step(self, final_iter):
        self.prior_mass.append(np.exp(-(final_iter+self._n_points/2.)/self._n_points))
        sum_all_L = 0.
        for point in self.points:
            sum_all_L += point.likelihood
        self.evidence.append(self.evidence[-1] + self.prior_mass[-1]*sum_all_L)
        self.weights.append(0.)

def nested_loop(N_iter, seed, *args):
    np.random.seed(seed)
    nest = nested(*args)
    i = 1
    while i<N_iter:
        start = time.time()
        nest.find_worst()
        nest.update_quantities(i)
        nest.substitute_worst()
        left = (time.time()-start)*(N_iter-i)
        if i%100 == 0:
            if left >= 3600.:
                #sys.stdout.flush()
                print(f'iteration n. {i}; expected time left: {left//3600} h {left//60%60:.0f} m {left%60:.0f} s.')
            elif left >= 60.:
                #sys.stdout.flush()
                print(f'iteration n. {i}; expected time left: {left//60} m {left%60:.0f} s.')
            else:
                #sys.stdout.flush()
                print(f'iteration n. {i}; expected time left: {left%60:.0f} s.')
        i += 1
        if 1. - nest.evidence[-2]/nest.evidence[-1] < 0.00001:
            break
    nest.final_step(i)
    print('simulation completed.')
    output_path = os.path.abspath('output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out = os.path.join(output_path, f'data_{seed}_{args[0]}_{N_iter}.csv')
    with open(out, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['iteration', 'prior mass', 'evidence'])
        j = 0
        for x, z in zip(nest.prior_mass, nest.evidence):
            writer.writerow([j, x, z])
            j += 1
    return i, nest.prior_mass[-1], nest.evidence[-1], nest.worst_L

def main():
    parser = argparse.ArgumentParser(description='Classical Nested Sampling Algorithm.')
    parser.add_argument('--N_iter', metavar='N_iter', type=int, dest='N_iter', help='sets maximum \#iterations.')
    parser.add_argument('--n_points', metavar='n_points', type=int, dest='n_points', help='sets \# points for simulation.')
    parser.add_argument('--dim', metavar='dim', type=int, dest='dim', help='dimension of the explored space.')
    parser.add_argument('--prior_range', metavar='prior_range', type=float, dest='prior_range', help='sets radius of the explored domain.')
    parser.add_argument('--MC_step', metavar='MC_step', type= float, dest='MC_step', help='sets step for MC evolution w.r.t. dimension of the space.\nNumbers <0.01 recommended.')
    parser.add_argument('--stoch_prior', dest='X_stoch', help='finds new values for prior mass stochastically;\ndefault: takes them according to exp(-i/N).')
    parser.add_argument('--trapezoid', dest='trapezoid', help='sets new weight using trapezoidal rule;\ndefault: takes difference btw consecutive X values.')
    parser.add_argument('--seed', metavar='seed', type=int, dest='seed', help='sets seed for simulation.')
    parser.add_argument('--n_runs', dest='n_runs', type=int, help='\#parallel runs of the simulation;\ndefault: 18.')
    parser.set_defaults(MC_step=0.005)
    parser.set_defaults(X_stoch=False)
    parser.set_defaults(trapezoid=False)
    parser.set_defaults(seed=1)
    parser.set_defaults(n_runs=18)
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
        print(f'total n. of iterations: {result[0]}; last prior mass value recorded: {result[1]};')
        print(f'value of evidence obtained: {result[2]}; last likelihood value: {result[3]}.\n')

if __name__ == '__main__':
    main()
