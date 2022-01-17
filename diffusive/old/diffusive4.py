import random
from math import floor

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

import new_MC
from new_MC import MCMC
import polar_nd
from polar_nd import polar_nd_init

"""
N_level = 600#300
N_max = 24000#12000
N_iter = 40000#20000
MC_step = 0.0015#0.001
"""
dim = 50
levels = 100#200
prior_range = 30.
quantile = 1. - np.e**-1
Q = 100

def gauss(point):
    r2 = (point*point).sum()
    return np.exp(-r2/2.)

def diffusive(dim, levels, prior_range, N_level, N_iter, N_max, MC_step, C, C1, beta, lam, seed):
    np.random.seed(seed)
    particle = polar_nd_init(dim, prior_range)
    L_recorded = []
    L_level = [np.exp(-prior_range**2/2.)]
    L_set = [gauss(particle)]
    current_level = 0
    level_visits = list(np.ones(levels+1))
    level_visits[0] += 1
    level_record = []
    #acceptance_record = []

    i = 1
    n_lk = 1
    while i < N_level:
        particle, current_level = MCMC(particle, dim, L_level[current_level]*MC_step, current_level, L_level, level_visits[current_level], level_visits[current_level-1], 0, 0, level_visits[0], 0, C=C, C1=C, beta=beta, lam=lam)
        L_set.append(gauss(particle))
        level_visits[0] += 1
        if i%Q == 0:
            L_recorded.append(L_set[-1])
            n_lk += 1
            print(f'creating level {0}, with L: {L_level[-1]}, final points: {n_lk}, L: {L_recorded[-1]}')
            #print(acceptance)
            #acceptance_record.append(acceptance)
            level_record.append(current_level)
        i += 1
    L_set.sort()
    L_quantile = L_set[floor(N_level*quantile)-1:]
    L_level.append(L_set[floor(len(L_set)*quantile)-1])
    n_tot = [[len(L_set[:floor(N_level*quantile)-1]), len(L_quantile)]]
    L_set = L_quantile

    j = 1
    while j < levels:
        trials = 0
        i = 0
        k = 0
        #print('--------------------------------------------------------------------------------------------')
        """
        while trials < N_level:
            if current_level == 0:
                particle, current_level = MCMC(particle, dim, np.sqrt(-2.*np.log(L_level[current_level]))*MC_step, current_level, L_level[-1:], level_visits[current_level-1], 0, n_tot[current_level][0], n_tot[current_level][1], level_visits[0], 0, C=C, C1=C, beta=beta, lam=lam)
            else:
                particle, current_level = MCMC(particle, dim, np.sqrt(-2.*np.log(L_level[current_level]))*MC_step, 0, [L_level[current_level]], 0, 0, n_tot[current_level][0], n_tot[current_level][1], level_visits[0], 0, C=C, C1=C, beta=beta, lam=lam)ù
            L_new = gauss(particle)
            if L_new > L_set[0]:
                #n_tot[j-1][0] += 1
                n_tot[j-1][1] += 1
            elif L_new > L_level[-1]:
                n_tot[j-1][0] += 1
            else:
                pass
            if (k%Q == 0):
                L_recorded.append(L_new)
                level_record.append(current_level)
                #print(acceptance)
                #acceptance_record.append(acceptance)
                n_lk += 1
            if trials == N_max*10:
                break
            trials += 1
        """
        #print(trials)
        while i < N_level:
            particle, current_level = MCMC(particle, dim, np.sqrt(-2.*np.log(L_level[current_level]))*MC_step, current_level, L_level,  level_visits[current_level], level_visits[current_level-1], n_tot[current_level-1][0], n_tot[current_level-1][1], level_visits[0], 0, C=C, C1=C, beta=beta, lam=lam)
            L_new = gauss(particle)
            level_visits[current_level] += 1
            """
            if L_new > L_set[0]:
                #n_tot[j-1][0] += 1
                n_tot[j-1][1] += 1
            elif L_new > L_level[-1]:
                n_tot[j-1][0] += 1
            else:
                pass
            """
            if (k%Q == 0):
                L_recorded.append(L_new)
                level_record.append(current_level)
                #print(acceptance)
                #acceptance_record.append(acceptance)
                n_lk += 1
                print(f'creating level {j}, with L: {L_level[-1]}, final points: {n_lk}, L: {L_recorded[-1]}')
            if L_new >= L_level[-1]:
                L_set.append(L_new)
                i += 1
            if n_lk == N_max:
                break
            k += 1
        #print(n_tot[j-1])
        L_set.sort()
        #L_level.append(L_set[floor(len(L_set)*quantile)-1])
        L_level.sort()
        L_quantile = L_set[floor(len(L_set)*quantile)-1:]
        L_level.append(L_set[floor(len(L_set)*quantile)-1])
        #print(len(L_set), len(L_quantile))
        print(n_tot[-1])
        n_tot.append([len(L_set[:floor(N_level*quantile)-1]), len(L_quantile)])
        L_set = L_quantile
        j += 1
        if n_lk == N_max:
            break
    it_unif = n_lk
    i = n_lk*Q
    n_tot.append([1,1])#([N_level, floor(N_level*(1-quantile))-1])
    X_swaths = [n_tot[0][1]/(n_tot[0][0]+n_tot[0][1])]
    for i in range(1, len(n_tot)):
        X_swaths.append((n_tot[i][1]/(n_tot[i][0]+n_tot[i][1]))*X_swaths[-1])
    X_swaths.append(X_swaths[-1]*np.e**-1)
    print(X_swaths)
    print(L_level)
    #print(len(n_tot))
    #print(len(L_level))
    print(level_visits)
    level_visits_old = level_visits
    level_visits = list(np.ones(levels+1))
    while n_lk < N_iter:
        particle, current_level = MCMC(particle, dim, np.sqrt(-2.*np.log(L_level[current_level]))*MC_step, current_level, L_level, level_visits[current_level], level_visits[current_level-1], n_tot[current_level][0], n_tot[current_level][1], level_visits[0], n_lk-it_unif, C=C, C1=C, beta=beta, lam=lam, level_build=False)
        L_new = gauss(particle)
        level_visits[current_level] += 1
        #for level in range(current_level+1):
            #level_visits[level] += 1
        if i%Q == 0:
            L_recorded.append(L_new)
            level_record.append(current_level)
            #acceptance_record.append(acceptance)
            n_lk += 1
            if n_lk%50 == 0:
                print(f'currently at level {current_level}, final points: {n_lk}, L: {L_recorded[-1]}')
                #print(acceptance)
        i += 1

    L_recorded.sort()
    level_visits += level_visits_old
    Z = [0.]
    X = [1.]
    w = []
    X_ratios = [X_swaths[i]/X_swaths[i-1] for i in range(1, len(X_swaths))]
    mean_ratios = sum(X_ratios)/len(X_ratios)
    var_ratios = sum([ratio**2 for ratio in X_ratios])/len(X_ratios)
    var_ratios -= mean_ratios**2
    dev_ratios = var_ratios**0.5
    #print(f'{mean_ratios} +- {var_ratios**0.5}')
    print(X_ratios)
    ratio_hist, bin_hist = np.histogram(X_ratios, 10)
    bin_hist = [(bin_hist[i]+bin_hist[i-1])/2 for i in range(1, len(bin_hist))]
    #plt.plot(bin_hist, ratio_hist, linestyle='', marker='*', color='black')
    #plt.show()
    #norm = [sum([np.exp(-j/level) for j in range(int(level))]) for level in level_visits]
    for like in L_recorded:
        j = 0
        i = 0
        while (i < len(L_level)-1) & (like > L_level[i]):
            i += 1
        #X.append((X_swaths[i]-X_swaths[i+1])*np.exp(-j/level_visits[i])/norm)#/level_visits[i]**2)
        if j == int(level_visits[i]):
            if i == len(L_level)-1:
                X.append(np.exp(-i)*((np.exp(-i-1))**(j/level_visits[i]*np.log(X_swaths[i]/X_swaths[i+1]))))
                #X.append(X_swaths[i]*(X_swaths[i+1]/X_swaths[i])**(j/level_visits[i]))
            else:
                X.append(np.exp(-i)*(np.exp(-i-1)**(j/level_visits[i]*np.log(X_swaths[i]/X_swaths[i+1]))-(np.exp(-i-2))**(1/level_visits[i+1]*np.log(X_swaths[i+1]/X_swaths[i+2]))))
                #X.append(X_swaths[i]*(X_swaths[i+1]/X_swaths[i])**(j/level_visits[i])-X_swaths[i+1])#(X_swaths[i+2]/X_swaths[i+1])**(1/level_visits[i+1])))
        else:
            X.append(np.exp(-i)*(np.exp(-i-1)**(j/level_visits[i]*np.log(X_swaths[i]/X_swaths[i+1]))-np.exp(-i-1)**((j+1)/level_visits[i]*np.log(X_swaths[i]/X_swaths[i+1]))))#/level_visits[i]**2)
            #X.append(X_swaths[i]*((X_swaths[i+1]/X_swaths[i])**(j/level_visits[i])-(X_swaths[i+1]/X_swaths[i])**(j+1/level_visits[i])))#/level_visits[i]**2)
        Z.append(Z[-1] + like*X[-1])
        j += 1
        #print(like, i, Z[-1])
    X.sort()#(reverse=True)
    return X, Z, level_record

if __name__ == '__main__':
    n_lev = [2000]
    n_it = [15000]
    MC = [0.005]
    C = [500]
    beta = [10]
    lam = [25]
    params_diff = [(dim, levels, prior_range, N_level, 5*N_level, 5*N_level, MC_step, N_level//10, N_level//10, bt, lm)
    for N_level in n_lev
    #for N_iter in n_it
    for MC_step in MC
    #for C_ in C
    for bt in beta
    for lm in lam]
    for param in params_diff:
        params = [(*param, i) for i in range(6)]
        start = time.time()
        with Pool() as pool:
            results = pool.starmap(diffusive, params)
        print(f'\#points for level: {params[0][3]}; \#iterations: {params[0][4]}; relative mc step: {params[0][6]};_{params[0][7]}_{params[0][9]}_{params[0][10]} exec time: {time.time()-start} Q {Q} levels {levels}')
        colours = ['black', 'red', 'blue', 'brown', 'green', 'orange', 'purple', 'grey']
        i = 0
        for X, Z, levelz in results:
            print(Z[-1])
            plt.plot(X, Z, linestyle='-', color=colours[i%8])
            plt.xscale('log')
            plt.yscale('log')
            i += 1
        plt.savefig(f'./graphs/{params[0][3]}_{params[0][4]}_{params[0][6]}_{params[0][7]}_{params[0][9]}_{params[0][10]}_{levels}.pdf')
        plt.clf()
        print('\n')
        i = 0
        for X, Z, levelz in results:
            plt.plot(range(len(levelz)), levelz, linestyle='-', color=colours[i%8])
            i += 1
        plt.savefig(f'./graphs/{params[0][3]}_{params[0][4]}_{params[0][6]}_{params[0][7]}_{params[0][9]}_{params[0][10]}_{levels}levels.pdf')
        plt.clf()
        
