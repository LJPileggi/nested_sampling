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
    parser.set_defaults(n_trials=12)
    args = parser.parse_args()
    return args

def data_reader(filename, datalen=12):
    folder = './output/'
    f_in = open(folder+filename, 'r', newline='')
    reader = csv.reader(f_in, delimiter=',')
    data = {}
    for i, line in enumerate(reader):
        result = list(map(float, line))
        if i%datalen == 0:
            data.update({result[0] : []})
        data[result[0]].append(result[2])
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
    return pars

def mean_var_plot(n_points, mean_list, var_list):
    plt.errorbar(n_points, mean_list, var_list**0.5, marker='*', linestyle='-', color='black')
    plt.axhline(y=-42+np.log10(7.25), color='red')
    plt.title('Logarithmic evidence vs $\#$initial points')
    plt.xscale('log')
    plt.xlabel('initial points')
    plt.ylabel('$log_{10}(Z)$')
    #plt.margins(0, -0.25)
    plt.savefig('logZ_vs_n_p.png')
    plt.clf()

    pars = power_fit(np.log10(var_list**0.5), np.log10(n_points))
    print(pars)
    plt.plot(n_points, var_list**0.5, linestyle='', marker='x', color='red')
    plt.plot(n_points, 10**pars[1]*n_points**pars[0], linestyle='-', color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Error on $log_{10}(Z)$ vs $\#$initial points')
    plt.xlabel('initial points')
    plt.ylabel('$\Delta log_{10}(Z)$')
    plt.savefig('d_logZ_n_p.png')

if __name__ == '__main__':
    args = parsing()
    dataset = data_reader(args.filename, args.n_trials)
    #print(dataset)
    mean, var = [], []
    for data in dataset.values():
        mean.append(mean_var(log10_data(data))[0])
        var.append(mean_var(log10_data(data))[1])
    mean, var = np.array(mean), np.array(var)
    n_points = np.array(list(dataset.keys()))
    mean_var_plot(n_points, mean, var)
