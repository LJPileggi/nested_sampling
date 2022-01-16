import random
from math import floor

import numpy as np

from polar_init import polar_nd_init

#namedtuple() converts dict to obj

def gauss(point):
    r2 = (point*point).sum()
    return np.exp(-r2/2.)

def proposal(x_t, len_step, dim):
    x_t1 = x_t + len_step*np.random.uniform(-1., 1., dim)
    return x_t1

class particle():
    def __init__(self, likelihood, dim, prior_range, params):
        self.theta = polar_nd_init(dim, prior_range)
        self.likelihood = likelihood(self.theta)
        self.current = 0
        self.L_levels = [likelihood(prior_range)]
        self.L_record = []
        self.level_visits = [0]
        self.relative_visits = []
        self._likelihood_func = likelihood
        self._dim = dim
        self._prior_range = prior_range
        self._params = params
        self._L_buffer = []
        self._iter = 0
        self._unif_iter = 0
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
        w = np.random.uniform(0., 1.)
        enforce = (self.level_visits[level-1] + self._params.C1)*(expected_visits(level) + self._params.C1)/(self.level_visits[level] + self._params.C1)/(expected_visits(level-1) + self._params.C1)
        if self._creating:
            ratio = weighting(level-1)/weighting(level)*(self.relative_visits[level-1][1] + self._params.C)/(self.relative_visits[level-1][0] + self._params.C*self._params.quantile**-1)*enforce**self._params.beta
        else:
            ratio = (self.relative_visits[level-1][1] + self._params.C)/(self.relative_visits[level-1][0] + self._params.C*self._params.quantile**-1)*enforce**self._params.beta
        if w > ratio:
            return True
        else:
            return False

    def MC_step(self):
        accept = False
        while not accept:
            theta_prop = proposal(self.theta, np.sqrt(-2.*np.log(self.L_levels[self.current]))*self._params-MC_step, self._dim)
            L_prop = self._likelihood_func(theta_prop)
            if L_prop > self.L_levels[current]:
                self.theta = theta_prop
                self.likelihood = L_prop
                accept = True
                while (self.likelihood > self.L_levels[self.current]) & (self.current != len(self.L_levels)-1):
                    if self.likelihood > self.L_levels[self.current+1]:
                        self.current += 1
                    else:
                        break
            elif self.current == 0:
                pass
            else:
                if level_switch(self.current):
                    self.theta = theta_prop
                    self.likelihood = L_prop
                    self.current -= 1
                    accept = True
