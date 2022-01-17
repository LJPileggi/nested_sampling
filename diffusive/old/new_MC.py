import numpy as np

def proposal(x_t, len_step, dim):
    x_t1 = x_t + len_step*np.random.uniform(-1., 1., dim)
    return x_t1

def likelihood(x):
    L = np.exp(-(x*x).sum()/2.)
    return L

def level_weight(level, max_levels, lam=25.):
    return np.exp((level-max_levels)/lam)

def level_count(level, max_levels, it_tot, it_unif):
    norm = 0.
    for i in range(max_levels):
        norm += level_weight(level, max_levels)
    """
    if it_unif == 0:
        return level_weight(level, max_levels)*it_tot/norm
    else:
        return it_unif/(max_levels+1)
    """
    return level_weight(level, max_levels)*(it_tot-it_unif)/norm + it_unif/(max_levels+1)

def level_switch(current_level, L_level, count_j, count_j_1, n_j, n_j_1, it, it_unif, C=1000., C1=1000, beta=1, lam=10, level_build=True):
    w = np.random.uniform(0., 1.)
    enforce = ((count_j_1 + C1)*(level_count(current_level, len(L_level)-1, it, it_unif) + C1)/(count_j + C1)/(level_count(current_level-1, len(L_level)-1, it, it_unif) + C1))**beta
    if level_build:
        ratio = level_weight(current_level-1, len(L_level)-1, lam=lam)/level_weight(current_level, len(L_level)-1, lam=lam) * (n_j_1+C*(1-np.e**-1))/(n_j+C)*enforce#**-1
    else:
        ratio = (n_j_1+C*(1-np.e**-1))/(n_j+C)*enforce#**-1
    if w > ratio:
        return True#, ratio
    else:
        return False#, ratio

def MCMC(theta, dim, len_step, current_level, L_level, count_j, count_j_1, n_j, n_j_1, it, it_unif, n=1, level_build=True, C=1000., C1=1000, beta=1, lam=10):
    theta_new = theta
    current = current_level
    i = 0
    acceptance = [True, 1.]
    while i < n:
        theta_prop = proposal(theta_new, len_step, dim)
        L_prop = likelihood(theta_prop)
        if L_prop > L_level[current_level]:
            theta_new = theta_prop
            i += 1
            while (L_prop > L_level[current]) & (current != len(L_level)-1):
                if L_prop > L_level[current+1]:
                    current += 1
                else:
                    break
        elif current_level == 0:
            pass
        else:
            acceptance = level_switch(current_level, L_level, count_j, count_j_1, n_j, n_j_1, it, it_unif, level_build=level_build, C=C, C1=C1, beta=beta, lam=lam)
            if acceptance:#[0]:
                theta_new = theta_prop
                i += 1
                current -= 1
    return theta_new, current#, acceptance[1]
