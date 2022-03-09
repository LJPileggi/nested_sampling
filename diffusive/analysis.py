import os
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parsing():
    parser = argparse.ArgumentParser(description='Analysis of results of Classical Nested Sampling.')
    parser.add_argument('--filename', metavar='filename', dest='filename', help='name of file to be analysed.')
    parser.add_argument('--n_trials', metavar='n_trials', type=int, dest='n_trials', help='\#trials on the same config.')
    parser.add_argument('--param', metavar='param', dest='param', help='parameter on the x axis. Choose between \'L_per_level\', \' lambda\', \'beta\', \'quantile\'. Set by default to \'L_per_level\'')
    parser.set_defaults(n_trials=18)
    parser.set_defaults(param='L_per_level')
    args = parser.parse_args()
    return args

def par_encod(param):
    if param == 'L_per_level':
        return 1
    elif param == 'lambda':
        return 3
    elif param == 'beta':
        return 4
    elif param == 'quantile':
        return 5
    else:
        raise ValueError('ValueError: parameter not present in model.')

def par_name_encod(param):
    if param == 'L_per_level':
        return 'initial points', 'n_p'
    elif param == 'lambda':
        return '$\lambda$', 'lam'
    elif param == 'beta':
        return '$\\beta$', 'b'
    elif param == 'quantile':
        return '$\\nu$', 'q'
    else:
        raise ValueError('ValueError: parameter not present in model.')


def data_reader(filename, param, datalen=18):
    folder = './output/'
    f_in = open(folder+filename, 'r', newline='')
    reader = csv.reader(f_in, delimiter=',')
    data = {}
    for i, line in enumerate(reader):
        if i == 0:
            pass
        else:
            result = list(map(float, line))
            if (i-1)%datalen == 0:
                data.update({result[par_encod(param)] : []})
            data[result[par_encod(param)]].append(result[6])
    return data

def log10_data(data):
    return np.log10(data)

def mean_var(values):
    mean, var = 0., 0.
    for value in values:
        mean += value/len(values)
        var += value**2/len(values)
    var -= mean**2
    return mean, var

def power_law(x, a, b):
    return a*x + b

def power_fit(dev, n_points):
    pars, covm = curve_fit(power_law, n_points, dev, (-0.5, 2))
    return pars, covm

def mean_var_plot(n_points, mean_list, var_list, param):
    if par_encod(param) == 1:
        plt.errorbar(n_points, mean_list, var_list, marker='*', linestyle='-', color='black')
        plt.axhline(y=-42+np.log10(7.25), color='red')
        plt.title('Logarithmic evidence vs '+par_name_encod(param)[0])
        plt.xscale('log')
        plt.xlabel(par_name_encod(param)[0])
        plt.ylabel('$log_{10}$(Z)')
        plt.savefig('logZ_vs_'+par_name_encod(param)[1]+'.png')
        plt.clf()

        pars, covm = power_fit(np.log10(var_list**0.5), np.log10(n_points))
        print(pars)
        print(covm[0][0]**0.5, covm[1][1]**0.5, covm[0][1]/covm[0][0]**0.5/covm[1][1]**0.5)
        plt.plot(n_points, var_list**0.5, linestyle='', marker='x', color='red')
        plt.plot(n_points, 10**pars[1]*n_points**pars[0], linestyle='-', color='black')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Error on $log_{10}$(Z) vs '+par_name_encod(param)[0])
        plt.xlabel(par_name_encod(param)[0])
        plt.ylabel('$\Delta$ $log_{10}$(Z)')
        plt.savefig('d_logZ_n_p'+par_name_encod(param)[1]+'.png')
    else:
        plt.errorbar(n_points, mean_list, var_list, marker='*', linestyle='-', color='black')
        plt.axhline(y=-42+np.log10(7.25), color='red')
        plt.title('Logarithmic evidence vs '+par_name_encod(param)[0])
        plt.xlabel(par_name_encod(param)[0])
        plt.ylabel('$log_{10}$(Z)')
        plt.savefig('logZ_vs_'+par_name_encod(param)[1]+'.png')
        plt.clf()

        #pars = power_fit(np.log10(var_list**0.5), np.log10(n_points))
        #print(pars)
        plt.plot(n_points, var_list**0.5, linestyle='', marker='x', color='red')
        #plt.plot(n_points, 10**pars[1]*n_points**pars[0], linestyle='-', color='black')
        plt.yscale('log')
        plt.title('Error on $log_{10}$(Z) vs '+par_name_encod(param)[0])
        plt.xlabel(par_name_encod(param)[0])
        plt.ylabel('$\Delta$ $log_{10}$(Z)')
        plt.savefig('d_logZ_'+par_name_encod(param)[1]+'.png')

if __name__ == '__main__':
    args = parsing()
    dataset = data_reader(args.filename, args.param, args.n_trials)
    #print(dataset)
    mean, var = [], []
    for data in dataset.values():
        mean.append(mean_var(log10_data(data))[0])
        var.append(mean_var(log10_data(data))[1])
    mean, var = np.array(mean), np.array(var)
    n_points = np.array(list(dataset.keys()))
    mean_var_plot(n_points, mean, var, args.param)
