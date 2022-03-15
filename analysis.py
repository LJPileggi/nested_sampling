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
    parser.set_defaults(param='normal')
    args = parser.parse_args()
    return args

class encoding():

    def __init__(self, param, alg, filename, n_trials):
        if alg not in ('classical', 'diffusive'):
            raise NotImplementedError('NotImplementedError: not yet implemented algorithm.')
        elif (alg == 'classical') & (param not in ('normal', 'X_stoch', 'trapezoid', 'time')):
            raise ValueError('ValueError: parameter not present in model.')
        elif (alg == 'diffusive') & (param not in ('normal', 'lambda', 'beta', 'quantile', 'time')):
            raise ValueError('ValueError: parameter not present in model.')
        self._param = param
        self._alg = alg
        self._filename = filename
        self._n_trials = n_trials
        if alg == 'classical':
            self._out_encod = 2
            if param in ('normal', 'X_stoch', 'trapezoid'):
                self._par_encod = 0
                self._par_name_encod = 'initial points'
            elif param == 'time':
                self._par_encod = 3
                self._par_name_encod = 'time'
        elif alg == 'diffusive':
            self._out_encod = 6
            if param == 'normal':
                self._par_encod = 1
                self._par_name_encod = 'Likelihood per level'
            elif param == 'lambda':
                self._par_encod = 3
                self._par_name_encod = '$\lambda$'
            elif param == 'beta':
                self._par_encod = 4
                self._par_name_encod = '$\\beta$'
            elif param == 'quantile':
                self._par_encod = 5
                self._par_name_encod = '$\\nu$'
            elif param == 'time':
                self._par_encod = 7
                self._par_name_encod = 'time'
        self._data_reader()

    def _data_reader(self):
        folder = './output/'+self._alg+'/'+self._param+'/'
        f_in = open(folder+self._filename, 'r', newline='')
        reader = csv.reader(f_in, delimiter=',')
        data = {}
        for i, line in enumerate(reader):
            if i == 0:
                pass
            else:
                result = list(map(float, line))
                if (i-1)%self._n_trials == 0:
                    data.update({result[self._par_encod] : []})
                data[result[self._par_encod]].append(result[self._out_encod])
        self._data = data

    def _mean_var(self):

        def mean_var(values):
            mean, var = 0., 0.
            for value in values:
                mean += value/len(values)
                var += value**2/len(values)
            var -= mean**2
            return mean, var

        def log10_data(data):
            return np.log10(data)

        mean, var = [], []
        for data in self._data.values():
            if self._param != 'time':
                mean.append(mean_var(log10_data(data))[0])
                var.append(mean_var(log10_data(data))[1])
            else:
                mean.append(mean_var(data)[0])
                var.append(mean_var(data)[1])
        self._mean, self._var = np.array(mean), np.array(var)
        self._n_points = np.array(list(self._data.keys()))

    def _file_encod(self):
        graph_path = os.path.abspath('./graphs/'+self._alg+'/'+self._param)
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        out = os.path.join(graph_path, f'{self._filename[:-4]}')
        err = os.path.join(graph_path, f'{self._filename[:-4]}_err')
        return out, err

    def mean_var_plot(self):

        def power_law(x, a, b):
            return a*x + b

        def power_fit(dev, n_points):
            pars, covm = curve_fit(power_law, n_points, dev, (-0.5, 2))
            return pars, covm

        self._mean_var()
        plt.errorbar(self._n_points, self._mean, self._var, marker='*', linestyle='-', color='black')
        plt.axhline(y=-42+np.log10(7.25), color='red')
        plt.title('Logarithmic evidence vs '+self._par_name_encod)
        plt.xscale('log') if self._param not in ('beta', 'quantile') else None
        plt.yscale('log') if self._param == 'time' else None
        plt.xlabel(self._par_name_encod)
        plt.ylabel('time [s]') if self._param == 'time' else plt.ylabel('$log_{10}$(Z)')
        plt.savefig(self._file_encod()[0])
        plt.clf()

        if self._param == 'normal':
            pars, covm = power_fit(np.log10(self._var**0.5), np.log10(self._n_points))
            print(f'model: Ax + B.\nA = {pars[0]:.2f} +- {covm[0][0]**0.5:.2f}, B = {pars[1]:.2f} +- {covm[1][1]**0.5:.2f}')
            print(f'correlation: {covm[0][1]/covm[0][0]**0.5/covm[1][1]**0.5:.3f}')
            plt.plot(self._n_points, self._var**0.5, linestyle='', marker='x', color='red')
            plt.plot(self._n_points, 10**pars[1]*self._n_points**pars[0], linestyle='-', color='black')
            plt.xscale('log')
            plt.yscale('log')
            plt.title('Error on $log_{10}$(Z) vs '+self._par_name_encod)
            plt.xlabel(self._par_name_encod)
            plt.ylabel('$\Delta$ $log_{10}$(Z)')
            plt.savefig(self._file_encod()[1])

if __name__ == '__main__':
    args = parsing()
    encod = encoding(args.param, args.algorithm, args.filename, args.n_trials)
    encod.mean_var_plot()
