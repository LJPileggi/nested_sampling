import numpy as np

def proposal(x_t, len_step, dim):
    x_t1 = x_t + len_step*np.random.uniform(-1., 1., dim)
    return x_t1

def MCMC(theta, likelihood, L_s, dim, len_step, n=100):
    theta_new = theta
    i = 0
    while (i <= n) | (likelihood(theta_new) < L_s):
        theta_prop = proposal(theta_new, len_step, dim)
        if likelihood(theta_prop) > L_s:
            theta_new = theta_prop
            i += 1
    return theta_new
