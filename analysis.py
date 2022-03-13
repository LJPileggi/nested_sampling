import os
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parsing():
    parser = argparse.ArgumentParser(description='Analysis of results of Classical and Diffusive Nested Sampling.')
    parser.add_argument('--algorithm', metavar='algorithm', dest='algorithm', help='chooses whose data to analyse.')
    parser.add_argument('--filename', metavar='filename', dest='filename', help='name of file to be analysed.')
    parser.add_argument('--n_trials', metavar='n_trials', type=int, dest='n_trials', help='\#trials on the same config.')
    parser.add_argument('--param', metavar='param', type=str, dest='param', help='parameter on the x axis. Choose between \'X_stoch\', \'trapezoid\' and \'time\' for classical, \' lambda\', \'beta\', \'quantile\' and \'time\' for diffusive n. s., or leave blank.')
    parser.set_defaults(n_trials=18)
    parser.set_defaults(param='points')
    args = parser.parse_args()
    return args
            
def data_reader(filename, param, alg, datalen=18):
    if param == 'points':
        folder = './output/'+alg+'/normal/'
    else:
        folder = './output/'+alg+'/'+param+'/'
    f_in = open(folder+filename, 'r', newline='')
    reader = csv.reader(f_in, delimiter=',')
    data = {}
    for i, line in enumerate(reader):
        if i == 0:
            pass
        else:
            result = list(map(float, line))
            if (i-1)%datalen == 0:
                data.update({result[par_encod(param, alg)] : []})
            data[result[par_encod(param, alg)]].append(result[res_encod(alg)])
    return data

def par_encod(param, alg):
    if alg == 'classical':
        if param in ('points', 'X_stoch', 'trapezoid'):
            return 0
        if param == 'time':
            return 3
        else:
            raise ValueError('ValueError: parameter not present in model.')
    elif alg == 'diffusive':
        if param == 'points':
            return 1
        elif param == 'lambda':
            return 3
        elif param == 'beta':
            return 4
        elif param == 'quantile':
            return 5
        elif param == 'time':
            return 7
        else:
            raise ValueError('ValueError: parameter not present in model.')
    else:
        raise NotImplementedError('NotImplementedError: not yet implemented algorithm.')

def res_encod(alg):
    if alg == 'classical':
        return 2
    elif alg == 'diffusive':
        return 6
    else:
        raise NotImplementedError('NotImplementedError: not yet implemented algorithm.')

def par_name_encod(param, alg):
    if alg == 'classical':
        if param in ('points', 'X_stoch', 'trapezoid'):
            return 'initial points'
        elif param == 'time':
            return 'time'
        else:
            raise ValueError('ValueError: parameter not present in model.')
    elif alg == 'diffusive':
        if param == 'points':
            return 'Likelihood per level'
        elif param == 'lambda':
            return '$\lambda$'
        elif param == 'beta':
            return '$\\beta$'
        elif param == 'quantile':
            return '$\\nu$'
        elif param == 'time':
            return 'time'
        else:
            raise ValueError('ValueError: parameter not present in model.')
    else:
        raise NotImplementedError('NotImplementedError: not yet implemented algorithm.')

def file_encod(param, alg, filename):
    clas_fold = './graphs/classical/'
    diff_fold = './graphs/diffusive/'
    if alg == 'classical':
        if param == 'points':
            graph_path = os.path.abspath(clas_fold+'normal')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        elif param == 'X_stoch':
            graph_path = os.path.abspath(clas_fold+'X_stoch')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        elif param == 'trapezoid':
            graph_path = os.path.abspath(clas_fold+'trapezoid')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        elif param == 'time':
            graph_path = os.path.abspath(clas_fold+'time')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        else:
            raise ValueError('ValueError: parameter not present in model.')
    elif alg == 'diffusive':
        if param == 'points':
            graph_path = os.path.abspath(diff_fold+'normal')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        elif param == 'lambda':
            graph_path = os.path.abspath(diff_fold+'lambda')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        elif param == 'beta':
            graph_path = os.path.abspath(diff_fold+'beta')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        elif param == 'quantile':
            graph_path = os.path.abspath(diff_fold+'quantile')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        elif param == 'time':
            graph_path = os.path.abspath(diff_fold+'time')
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            out = os.path.join(graph_path, f'{filename[:-4]}')
            err = os.path.join(graph_path, f'{filename[:-4]}_err')
        else:
            raise ValueError('ValueError: parameter not present in model.')
    else:
        raise NotImplementedError('NotImplementedError: not yet implemented algorithm.')
    return out, err

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

def mean_var_plot(n_points, mean_list, var_list, param, alg, filename):
    plt.errorbar(n_points, mean_list, var_list, marker='*', linestyle='-', color='black')
    plt.axhline(y=-42+np.log10(7.25), color='red')
    plt.title('Logarithmic evidence vs '+par_name_encod(param, alg))
    plt.xscale('log') if param not in ('beta', 'quantile') else None
    plt.yscale('log') if param == 'time' else None
    plt.xlabel(par_name_encod(param, alg))
    plt.ylabel('time [s]') if param == 'time' else plt.ylabel('$log_{10}$(Z)')
    plt.savefig(file_encod(param, alg, filename)[0])
    plt.clf()

    if param == 'points':
        pars, covm = power_fit(np.log10(var_list**0.5), np.log10(n_points))
        print(f'model: Ax + B.\nA = {pars[0]:.2f} +- {covm[0][0]**0.5:.2f}, B = {pars[1]:.2f} +- {covm[1][1]**0.5:.2f}')
        print(f'correlation: {covm[0][1]/covm[0][0]**0.5/covm[1][1]**0.5:.3f}')
        plt.plot(n_points, var_list**0.5, linestyle='', marker='x', color='red')
        plt.plot(n_points, 10**pars[1]*n_points**pars[0], linestyle='-', color='black')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Error on $log_{10}$(Z) vs '+par_name_encod(param, alg))
        plt.xlabel(par_name_encod(param, alg))
        plt.ylabel('$\Delta$ $log_{10}$(Z)')
        plt.savefig(file_encod(param, alg, filename)[1])

if __name__ == '__main__':
    args = parsing()
    dataset = data_reader(args.filename, args.param, args.algorithm, args.n_trials)
    mean, var = [], []
    for data in dataset.values():
        if args.param != 'time':
            mean.append(mean_var(log10_data(data))[0])
            var.append(mean_var(log10_data(data))[1])
        else:
            mean.append(mean_var(data)[0])
            var.append(mean_var(data)[1])
    mean, var = np.array(mean), np.array(var)
    n_points = np.array(list(dataset.keys()))
    mean_var_plot(n_points, mean, var, args.param, args.algorithm, args.filename)
