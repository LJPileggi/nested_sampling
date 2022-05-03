import os
import sys
import csv
import random
import math
from math import floor

import numpy as np
import matplotlib.pyplot as plt
import time

from .utilities.polar_init import polar_nd_init
from .utilities.MCMC import MCMC
from .utilities.time_buffer import pipeline

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
    def __init__(self, N_iter, n_points, dim, prior_range, MC_step=0.005, X_stoch=False, trapezoid=False):
        self.n_points = n_points
        self._dim = dim
        self._prior_range = prior_range
        self._MC_step = MC_step
        self._X_stoch = X_stoch
        self._trapezoid = trapezoid
        self.N_iter = N_iter
        self.points = []
        i = 0
        while i<self.n_points:
            theta = polar_nd_init(dim, prior_range)
            self.points.append(point(theta, gauss(theta)))
            i += 1
        self.evidence = [0.]
        if not trapezoid:
            self.prior_mass = [1.]
        else:
            self.prior_mass = [2.-np.exp(-1./n_points), np.exp(-1./n_points)]
        self.weights = []
        self.worst_L_series = [0.]
        self.worst_idx = 0
        self.worst_L = 0.
        self.time = 0

    def find_worst(self):
        worst, L_w = 0, self.points[0].likelihood
        i = 1
        while i<self.n_points:
            if self.points[i].likelihood < L_w:
                worst, L_w = i, self.points[i].likelihood
            i += 1
        self.worst_idx, self.worst_L = worst, L_w
        self.worst_L_series.append(self.worst_L)

    def update_quantities(self, iter_step):
        if not self._X_stoch:
            self.prior_mass.append(np.exp(-iter_step/self.n_points))
        else:
            self.prior_mass.append(self.prior_mass[-1]*np.random.uniform(0., 1.)**(1./(self.n_points-1)))
        if not self._trapezoid:
            self.weights.append(self.prior_mass[iter_step-1]-self.prior_mass[iter_step])
        else:
            self.weights.append((self.prior_mass[iter_step-1]-self.prior_mass[iter_step+1])/2.)
        self.evidence.append(self.evidence[-1] + self.worst_L*self.weights[-1])

    def substitute_worst(self):
        idx_new = random.choice(range(self.n_points))
        while idx_new == self.worst_idx:
            idx_new = random.choice(range(self.n_points))
        new_point = point(self.points[idx_new].theta, self.points[idx_new].likelihood)
        new_point.MC_evolution(gauss, self.worst_L, self._dim, np.sqrt(-2.*np.log(self.worst_L))*self._MC_step)
        self.points[self.worst_idx] = new_point

    def final_step(self, final_iter):
        self.prior_mass.append(np.exp(-(final_iter+self.n_points/2.)/self.n_points))
        sum_all_L = 0.
        for point in self.points:
            sum_all_L += point.likelihood
        self.evidence.append(self.evidence[-1] + self.prior_mass[-1]*sum_all_L)
        self.weights.append(0.)

def nested_loop(N_iter, seed, *args):
    np.random.seed(seed)
    init_time_start = time.time()
    pipe = pipeline()
    trial = polar_nd_init(args[1], args[2])
    init_time = (time.time() - init_time_start)*args[0]
    if not args[-1]:
        if init_time >= 3600.:
            print(f'process {os.getpid()}; initialising particles. Expected time for initialisation: '
                  f'{init_time//3600:.0f} h {init_time//60%60:.0f} m {init_time%60:.0f} s.')
        elif init_time >= 60.:
            print(f'process {os.getpid()}; initialising particles. Expected time for initialisation: {init_time//60} m {init_time%60:.0f} s.')
        else:
            print(f'process {os.getpid()}; initialising particles. Expected time for initialisation: {init_time%60:.0f} s.')
    nest = nested(N_iter, *args[:-1])
    i = 1
    while i<N_iter:
        start = time.time()
        nest.find_worst()
        nest.update_quantities(i)
        nest.substitute_worst()
        it = (time.time()-start)
        pipe.update(it)
        left = pipe.average()*(N_iter-i)
        if (i%100 == 0) & (not args[-1]):
            if left >= 3600.:
                print(f'process {os.getpid()}; iteration n. {i}, with worst L: {nest.worst_L_series[-1]}; expected time left: '
                      f'{left//3600:.0f} h {left//60%60:.0f} m {left%60:.0f} s.')
            elif left >= 60.:
                print(f'process {os.getpid()}; iteration n. {i}, with worst L: {nest.worst_L_series[-1]}; expected time left: {left//60} m {left%60:.0f} s.')
            else:
                print(f'process {os.getpid()}; iteration n. {i}, with worst L: {nest.worst_L_series[-1]}; expected time left: {left%60:.1f} s.')
        i += 1
        #if 1. - nest.evidence[-2]/nest.evidence[-1] < 0.00001:
            #break
    #nest.final_step(i)
    nest.time = time.time()-init_time_start
    if nest.time >= 3600:
        print(f'process {os.getpid()}; simulation completed. \#points: {args[0]}; time taken: {nest.time//3600:.0f} h '
              f'{nest.time//60%60:.0f} m {nest.time%60:.0f} s.')
    elif nest.time >= 60:
        print(f'process {os.getpid()}; simulation completed. \#points: {args[0]}; time taken: {nest.time//60:.0f} m {nest.time%60:.0f} s.')
    else:
        print(f'process {os.getpid()}; simulation completed. \#points: {args[0]}; time taken: {nest.time:.1f} s.')
    print(f'process {os.getpid()}; \#iterations: {len(nest.weights)}; last prior mass: {nest.prior_mass[-1]};')
    print(f'process {os.getpid()}; evidence: {nest.evidence[-1]}; last likelihood value: {nest.worst_L}.\n')
    return nest
