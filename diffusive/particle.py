import random
from math import floor
import time

import numpy as np

from polar_init import polar_nd_init

#namedtuple() converts dict to obj

def gauss(point):
    r2 = (point*point).sum()
    return np.exp(-r2/2.)

def proposal(x_t, len_step, dim):
    x_t1 = x_t + len_step*np.random.uniform(-1., 1., dim)
    return x_t1

"""
class params:
methods:
- C
- C1
- lam
- beta
- quantile
- MC_step
- max_level
- L_per_level
- max_recorded_points
- record_step
"""

class particle():
    def __init__(self, likelihood, dim, prior_range, params):
        self.theta = polar_nd_init(dim, prior_range)
        self.likelihood = likelihood(self.theta)
        self.current = 0
        self.L_levels = [0.]
        self.L_record = [0.]
        self.level_visits = [1]
        self.relative_visits = []
        self.iter = 1
        self.level_record = [0]
        self.prior_mass = []
        self.evidence = [0.]
        self._likelihood_func = likelihood
        self._dim = dim
        self._prior_range = prior_range
        self._params = params
        self._L_buffer = []
        self._swaths = []
        self._unif_iter = 0
        self._level_visits_old = []
        self._creating = True

    def weighting(self, level):
        return np.exp((level-len(self.L_levels)+1)/self._params.lam)

    def expected_visits(self, level):
        norm = 0.
        i = 0
        while i < len(self._L_levels):
            norm += self.weighting(i)
        return weighting(level)*(self._iter-self._unif_iter)/norm + self._unif_iter/len(self._L_levels)

    def level_switch(self, level):
        if level == 0:
            return False
        w = np.random.uniform(0., 1.)
        enforce = (self.level_visits[level-1] + self._params.C1)*(expected_visits(level) + self._params.C1)/(self.level_visits[level] + self._params.C1)/(expected_visits(level-1) + self._params.C1)
        if self._creating:
            ratio = weighting(level-1)/weighting(level)*self._params.quantile**-1/(1.-self._params.quantile**-1)*enforce**self._params.beta
        else:
            ratio = self._params.quantile**-1/(1.-self._params.quantile**-1)*enforce**self._params.beta
        if w > ratio:
            return True
        else:
            return False

    def MC_step(self):
        accept = False
        while not accept:
            theta_prop = proposal(self.theta, np.sqrt(-2.*np.log(self.L_levels[self.current]))*self._params.MC_step, self._dim)
            L_prop = self._likelihood_func(theta_prop)
            if L_prop > self.L_levels[current] & ((theta*theta).sum() > self._prior_range):
                self.theta = theta_prop
                self.likelihood = L_prop
                accept = True
                while (self.likelihood > self.L_levels[self.current]) & (self.current != len(self.L_levels)-1):
                    if self.likelihood > self.L_levels[self.current+1]:
                        self.current += 1
                    else:
                        break
            else:
                if level_switch(self.current) & ((theta*theta).sum() > self._prior_range):
                    self.theta = theta_prop
                    self.likelihood = L_prop
                    self.current -= 1
                    accept = True

    def create_level(self, new_level):
        start = time.time()
        level_points = len(self._L_buffer)
        while level_points <= self._params.L_per_level:
            self.iter += 1
            if self.iter%self._params.record_step > self._params.max_recorded_points:
                break
            MC_step()
            if self.likelihood >= self.L_levels[-1]:
                self._L_buffer.append(self.likelihood)
                level_points += 1
            self.level_visits[self.current] += 1
            if new_level > 0:
                if self.likelihood > self.L_levels[-1]:
                    self.relative_visits[new_level-1][1] += 1./weighting(new_level)
                elif self.likelihood > self.L_levels[-2]:
                    self.relative_visits[new_level-1][0] += 1./weighting(new_level-1)
                else:
                    pass
            if self.iter%self._params.record_step == 0:
                self.L_record.append(self.likelihood)
                self.level_record.append(self.current)
        self._L_buffer.sort()
        quant = floor(len(self._L_buffer)*(1-self._params.quantile))-1
        self.L_levels.append(self._L_buffer[quant])
        self.L_levels.sort()
        self.relative_visits.append([quant, len(self._L_buffer)*self._params.quantile])
        self._L_buffer = self._L_buffer[quant:]
        self.level_visits.append(1)

    def create_all_levels(self):
        while len(self.L_levels) <= self._params.max_level:
            create_level(self.L_levels)
            if self.iter%self._params.record_step > self._params.max_recorded_points:
                break
        self._level_visits_old = self.level_visits
        self._creating = False
        time_left = (time.time()-start)*(self._params.max_level-new_level)
        if time_left >= 3600.:
            print(f'Created level {new_level}. Expected time to finish creating levels: {time_left//3600} h {time_left//60%60:.0f} m {time_left%60:.0f} s.')
        elif time_left >= 60.:
            print(f'Created level {new_level}. Expected time to finish creating levels: {time_left//60} m {time_left%60:.0f} s.')
        else:
            print(f'Created level {new_level}. Expected time to finish creating levels: {time_left:.0f} s.')
            

    def explore_levels(self):
        while True:
            start = time.time()
            self.level_visits = list(np.ones(len(self.level_visits)))
            self.iter += 1
            self._unif_iter += 1
            if self.iter%self._params.record_step > self._params.max_recorded_points:
                break
            MC_step()
            self.level_visits[self.current] += 1
            if self.iter%self._params.record_step == 0:
                self.L_record.append(self.likelihood)
                self.level_record.append(self.current)
                time_left = (time.time()-start)*(self._params.max_recorded_points*self._params.record_step - self.iter)
            if time_left >= 3600.:
                print(f'{self.iter%self._params.record_step:.0f}th value collected. Expected time to finish: {time_left//3600} h {time_left//60%60:.0f} m {time_left%60:.0f} s.')
            elif time_left >= 60.:
                print(f'{self.iter%self._params.record_step:.0f}th value collected. Expected time to finish: {time_left//60} m {time_left%60:.0f} s.')
            else:
                print(f'{self.iter%self._params.record_step:.0f}th value collected. Expected time to finish: {time_left:.0f} s.')    
        self.level_visits += self._level_visits_old

    def find_evidence(self):
        self.L_record.sort()
        self._swaths.append(self.relative_visits[0][1]/(self.relative_visits[0][1]+self.relative_visits[0][0]))
        for i in range(1, len(self.relative_visits)):
            self._swaths.append(self._swaths[-1]*self.relative_visits[i][1]/(self.relative_visits[i][1]+self.relative_visits[i][0]))
        for like in self.L_record:
            i, j = 0, 0
            while (i < len(self.L_levels)-1) & (like > self.L_levels[i]):
                i += 1
            if j == int(self.level_visits[i]):
                if i == len(self.L_levels)-1:
                    self.prior_mass.append(self._params.quantile*np.exp(-i-1)**(j/self.level_visits[i]*np.log(self.swaths[i]/self.swaths[i+1])))
                else:
                    self.prior_mass.append(self._params.quantile*(np.exp(-i-1)**(j/self.level_visits[i]*np.log(self.swaths[i]/self.swaths[i+1]))- np.exp(-i-2)**(1/self.level_visits[i+1]*np.log(self.swaths[i+1]/self.swaths[i+2]))))
            else:
                self.prior_mass.append(self._params.quantile*(np.exp(-i-1)**(j/self.level_visits[i]*np.log(self.swaths[i]/self.swaths[i+1])) - np.exp(-i-1)**((j+1)/self.level_visits[i]*np.log(self.swaths[i]/self.swaths[i+1]))))
            self.evidence.append(self.evidence[-1] + like*self.prior_mass[-1])
            j += 1

def diffusive_loop(seed, likelihood, dim, prior_range, params):
    np.random.seed(seed)
    part = particle(likelihood, dim, prior_range, params)
    part.create_all_levels()
    part.explore_levels()
    part.find_evidence()
    print('Simulation completed.')
    output_path = os.path.abspath('output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    out = os.path.join(output_path, f'data_{seed}_{params.max_level}_{params.L_per_level}_l{params.lam}b{params.beta}Q{params.quantile}.csv')
    with open(out, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['iteration', 'level', 'prior mass', 'likelihood', 'evidence'])
        j = 0
        for x, y, z, t in zip(part.level_record, part.prior_mass, part._L_record, part.evidence):
            writer.writerow([j, x, y, z, t])
            j += 1
    return part
