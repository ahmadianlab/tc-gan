from __future__ import print_function

import numpy


def weight(x, J, delta, sigma, rs):
    z = rs.uniform(size=(len(x), len(x)))
    return numpy.exp(-(x - x.T)**2/(2 * sigma**2)) * (J + delta * z)


def generate_weight(N, J, delta, sigma, seed=None):
    J = numpy.asarray(J)
    delta = numpy.asarray(delta)
    sigma = numpy.asarray(sigma)
    rs = numpy.random.RandomState(seed)
    x = numpy.linspace(-1, 1, N).reshape((1, -1))
    W = numpy.empty((2 * N, 2 * N))
    W[:N, :N] = weight(x, J[0, 0], delta[0, 0], sigma[0, 0], rs)
    W[N:, :N] = weight(x, J[1, 0], delta[1, 0], sigma[1, 0], rs)
    W[:N, N:] = weight(x, -J[0, 1], delta[0, 1], sigma[0, 1], rs)
    W[N:, N:] = weight(x, -J[1, 1], delta[1, 1], sigma[1, 1], rs)
    return W
