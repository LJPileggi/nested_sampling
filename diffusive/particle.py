import os
import random
from math import floor
import time
import csv

import numpy as np
from collections import namedtuple
from types import SimpleNamespace

from polar_init import polar_nd_init

#namedtuple() converts dict to obj

def gauss(point):
    r2 = (point*point).sum()
    return np.exp(-r2/2.)

def proposal(x_t, len_step, dim):
    x_t1 = x_t + len_step*np.random.uniform(-1., 1., dim)
    return x_t1

class particle():
    def __init__(self, likelihood, dim, prior_range, params, no_search):
        self.theta = polar_nd_init(dim, prior_range)
        self.likelihood = likelihood(self.theta)
        self.current = 0
        self._likelihood_func = likelihood
        self.L_levels = [self._likelihood_func(np.array([prior_range]))]
        self.L_record = [0.]
        self.level_visits = [1]
        self.relative_visits = []
        self.iter = 1
        self.level_record = [0]
        self.prior_mass = []
        self.evidence = [0.]
        self.levels_finished = 0
        self._dim = dim
        self._prior_range = prior_range
        self.params = params
        self._L_buffer = []
        self._swaths = [1.]
        self._unif_iter = 0
        self._level_visits_old = []
        self._creating = True
        self._no_search = no_search

    def weighting(self, level):
        return np.exp((level-len(self.L_levels)+1)/self.params.lam)

    def expected_visits(self, level):
        norm = 0.
        i = 0
        while i < len(self.L_levels):
            norm += self.weighting(i)
            i += 1
        return self.weighting(level)*(self.iter-self._unif_iter)/norm + self._unif_iter/len(self.L_levels)

    def level_switch(self, level):
        w = np.random.uniform(0., 1.)
        enforce = (self.level_visits[level-1] + self.params.C1)*(self.expected_visits(level) + self.params.C1)/(self.level_visits[level] + self.params.C1)/(self.expected_visits(level-1) + self.params.C1)
        if self._creating:
            ratio = self.weighting(level-1)/self.weighting(level)*(self.relative_visits[level-1][0]+self.params.C1*(1.-self.params.quantile))/(self.relative_visits[level-1][1]+self.params.C1)*enforce**self.params.beta
        else:
            ratio = (self.relative_visits[level-1][0]+self.params.C1*(1.-self.params.quantile))/(self.relative_visits[level-1][1]+self.params.C1)*enforce**self.params.beta
        if w > ratio:
            return True
        else:
            return False

    def MC_step(self):
        accept = False
        while not accept:
            theta_prop = proposal(self.theta, np.sqrt(-2.*np.log(self.L_levels[self.current]))*self.params.MC_step, self._dim)
            L_prop = self._likelihood_func(theta_prop)
            if L_prop > self.L_levels[self.current]:
                self.theta = theta_prop
                self.likelihood = L_prop
                accept = True
                i = 0
                while (self.likelihood > self.L_levels[self.current]) & (self.current != len(self.L_levels)-1):
                    if self.likelihood > self.L_levels[self.current+1]:
                        self.current += 1
                    else:
                        break
            elif self.current == 0:
                pass
            else:
                if self.level_switch(self.current) & ((self.theta*self.theta).sum() < self._prior_range**2):
                    self.theta = theta_prop
                    self.likelihood = L_prop
                    self.current -= 1
                    accept = True
             #print(self.likelihood, self.L_levels[self.current])

    def create_level(self, new_level):
        start = time.time()
        i = 0
        while i <= self.params.L_per_level:#len(self._L_buffer)
            self.iter += 1
            if self.iter%self.params.record_step > self.params.max_recorded_points:
                break
            self.MC_step()
            if self.likelihood >= self.L_levels[-1]:
                self._L_buffer.append(self.likelihood)
                i += 1
            self.level_visits[self.current] += 1
            if new_level > 0:
                if self.likelihood > self.L_levels[-1]:
                    self.relative_visits[new_level-1][1] += 1#./self.weighting(new_level)
                elif self.likelihood > self.L_levels[-2]:
                    self.relative_visits[new_level-1][0] += 1#./self.weighting(new_level-1)
                else:
                    pass
            if self.iter%self.params.record_step == 0:
                self.L_record.append(self.likelihood)
                self.level_record.append(self.current)
                #print(f'Creating level {new_level}; L_{new_level}: {self.L_levels[-1]}; currently at {self.likelihood}')
        self._L_buffer.sort()
        quant = floor(len(self._L_buffer)*(1-self.params.quantile))-1
        self.L_levels.append(self._L_buffer[quant])
        self.L_levels.sort()
        self.relative_visits.append([quant, len(self._L_buffer)*self.params.quantile])
        #print(self._L_buffer)
        self._L_buffer = self._L_buffer[quant:]
        #print(self._L_buffer)
        self.level_visits.append(1)
        time_left = (time.time()-start)*(self.params.max_level-new_level)
        if self._no_search:
            if time_left >= 3600.:
                print(f'process {os.getpid()}; created level {new_level}. Expected time to finish creating levels: {time_left//3600:.0f} h {time_left//60%60:.0f} m {time_left%60:.0f} s.')
            elif time_left >= 60.:
                print(f'process {os.getpid()}; created level {new_level}. Expected time to finish creating levels: {time_left//60} m {time_left%60:.0f} s.')
            else:
                print(f'process {os.getpid()}; created level {new_level}. Expected time to finish creating levels: {time_left:.1f} s.')


    def create_all_levels(self):
        while len(self.L_levels) <= self.params.max_level:
            self.create_level(len(self.L_levels)-1)
            if self.iter%self.params.record_step > self.params.max_recorded_points:
                break
        self._level_visits_old = self.level_visits
        self._creating = False            

    def explore_levels(self):
        self.level_visits = list(np.ones(len(self.level_visits)))
        while True:
            start = time.time()
            self.iter += 1
            self._unif_iter += 1
            if self.iter//self.params.record_step > self.params.max_recorded_points:
                break
            self.MC_step()
            self.level_visits[self.current] += 1
            if self.iter%self.params.record_step == 0:
                self.L_record.append(self.likelihood)
                self.level_record.append(self.current)
                time_left = (time.time()-start)*(self.params.max_recorded_points*self.params.record_step - self.iter)
                if (self.iter//self.params.record_step%100 == 0) & self._no_search:
                    if time_left >= 3600.:
                        print(f'process {os.getpid()}; {self.iter//self.params.record_step:.0f}th value collected. Currently at level {self.current} with L {self.likelihood}. Expected time to finish: {time_left//3600:.0f} h {time_left//60%60:.0f} m {time_left%60:.0f} s.')
                    elif time_left >= 60.:
                        print(f'process {os.getpid()}; {self.iter//self.params.record_step:.0f}th value collected. Currently at level {self.current} with L {self.likelihood}. Expected time to finish: {time_left//60} m {time_left%60:.0f} s.')
                    else:
                        print(f'process {os.getpid()}; {self.iter//self.params.record_step:.0f}th value collected. Currently at level {self.current} with L {self.likelihood}. Expected time to finish: {time_left:.1f} s.')    
        self.level_visits += self._level_visits_old
        self.levels_finished = self.iter//self.params.record_step

    def find_evidence(self):
        self.L_record.sort()
        self._swaths.append(self.relative_visits[0][1]/(self.relative_visits[0][1]+self.relative_visits[0][0]))
        for i in range(1, len(self.relative_visits)):
            self._swaths.append(self._swaths[-1]*self.relative_visits[i][1]/(self.relative_visits[i][1]+self.relative_visits[i][0]))
        self._swaths.append(self._swaths[-1]*self.params.quantile)
        for like in self.L_record:
            i, j = 0, 0
            while (i < len(self.L_levels)-1) & (like > self.L_levels[i]):
                i += 1
            if j == int(self.level_visits[i]):
                if i == len(self.L_levels)-1:
                    self.prior_mass.append(self.params.quantile**i*np.exp(-i-1)**(j/self.level_visits[i]*np.log(self._swaths[i]/self._swaths[i+1])))
                else:
                    self.prior_mass.append(self.params.quantile**i*(np.exp(-i-1)**(j/self.level_visits[i]*np.log(self._swaths[i]/self._swaths[i+1]))- np.exp(-i-2)**(1/self.level_visits[i+1]*np.log(self._swaths[i+1]/self._swaths[i+2]))))
            else:
                self.prior_mass.append(self.params.quantile**i*(np.exp(-i-1)**(j/self.level_visits[i]*np.log(self._swaths[i]/self._swaths[i+1])) - np.exp(-i-1)**((j+1)/self.level_visits[i]*np.log(self._swaths[i]/self._swaths[i+1]))))
            self.evidence.append(self.evidence[-1] + like*self.prior_mass[-1])
            j += 1

def diffusive_loop(seed, likelihood, dim, prior_range, params, no_search):
    start = time.time()
    np.random.seed(seed)
    params = SimpleNamespace(**params)
    part = particle(likelihood, dim, prior_range, params, no_search)
    part.create_all_levels()
    part.explore_levels()
    part.find_evidence()
    print(f'process {os.getpid()}; simulation completed. \#points per level: {params.L_per_level};')
    print(f'lambda: {params.lam}; beta: {params.beta};')
    print(f'quantile: {params.quantile}; evidence: {part.evidence[-1]}; time taken: {time.time()-start}\n')
    output_path = os.path.abspath('output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out = os.path.join(output_path, f'data_{seed}_{params.max_level}_{params.L_per_level}_l{params.lam}b{params.beta}Q{params.quantile:.4f}.csv')
    """
    with open(out, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['max_level', 'L_per_level', 'max_points', 'C1', 'lam', 'beta', 'quantile'])
        writer.writerow([params.max_level, params.L_per_level, params.max_recorded_points, params.C1, params.lam, params.beta, params.quantile])
        writer.writerow(['iteration', 'level', 'prior mass', 'likelihood', 'evidence'])
        j = 0
        for x, y, z, t in zip(part.level_record, part.prior_mass, part.L_record, part.evidence):
            writer.writerow([j, x, y, z, t])
            j += 1
    """
    return part
