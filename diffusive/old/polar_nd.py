import math

import numpy as np

def multiply(*args):
    return np.prod(args)

def polar_nd_init(dim, radius):
    U = np.random.uniform(0., 1.)
    R = radius*U**(1./dim)
    phis = np.random.uniform(0., np.pi, dim-2)
    theta = np.random.uniform(0., 2*np.pi)
    coord = [np.cos(phis[0])]
    i = 1
    while i<dim-2:
        coord.append(np.cos(phis[i]))
        coord[-1] *= multiply(np.sin(phis[0:i]))
        i += 1
    coord.append(np.cos(theta))
    coord[-1] *= multiply(np.sin(phis))
    coord.append(np.sin(theta))
    coord[-1] *= multiply(np.sin(phis))
    coord = np.array(coord)*R
    return coord
