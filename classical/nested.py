import os
import sys
import csv
import random
import math
from math import floor

import numpy as np
import matplotlib.pyplot as plt
import time

from polar_init import polar_nd_init
from MCMC import MCMC

def gauss(point):
    r2 = (point*point).sum()
    return np.exp(-r2/2.)

class point():
    def __init__(self, theta, likelihood):
        self.theta = theta
        self.likelihood = likelihood

    def MC_evolution(self, likelihood, L_s, dim, len_step):
        self.theta = MCMC(self.theta, likelihood, L_s, dim, len_step)
        self.likelihood = likelihood(self.theta)

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
        new_point = self.points[idx_new]
        new_point.MC_evolution(gauss, self.worst_L, self._dim, np.sqrt(-2.*np.log(self.worst_L))*self._MC_step)
        self.points[self.worst_idx] = new_point

    def final_step(self, final_iter):
        self.prior_mass.append(np.exp(-(final_iter+self._n_points/2.)/self._n_points))
        sum_all_L = 0.
        for point in self.points:
            sum_all_L += point.likelihood
        self.evidence.append(self.evidence[-1] + self.prior_mass[-1]*sum_all_L)
        self.weights.append(0.)

def nested_loop(N_iter, seed, *args):
    np.random.seed(seed)
    init_time_start = time.time()
    trial = polar_nd_init(args[2], args[3])
    init_time = (time.time() - init_time_start)*args[1]
    if init_time >= 3600.:
        print(f'Initialising particles. Expected time for initialisation: {init_time//3600} h {init_time//60%60:.0f} m {init_time%60:.0f} s.')
    elif init_time >= 60.:
        print(f'Initialising particles. Expected time for initialisation: {init_time//60} m {init_time%60:.0f} s.')
    else:
        print(f'Initialising particles. Expected time for initialisation: {init_time%60:.0f} s.')
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
                print(f'iteration n. {i}; expected time left: {left//3600} h {left//60%60:.0f} m {left%60:.0f} s.')
            elif left >= 60.:
                print(f'iteration n. {i}; expected time left: {left//60} m {left%60:.0f} s.')
            else:
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

