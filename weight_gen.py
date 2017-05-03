from __future__ import print_function

import numpy


def weight(x, J, delta, sigma, z):
    """
    Generate a single block of connectivity matrix.
    """
    return numpy.exp(-(x - x.T)**2/(2 * sigma**2)) * (J + delta * z)


def generate_weight(N, J, delta, sigma, z):
    """
    Generate 2N-by-2N connectivity matrix.
    """
    J = numpy.asarray(J)
    delta = numpy.asarray(delta)
    sigma = numpy.asarray(sigma)
    x = numpy.linspace(-0.5, 0.5, N).reshape((1, -1))
    W = numpy.empty((2 * N, 2 * N))
    W[:N, :N] = weight(x, J[0, 0], delta[0, 0], sigma[0, 0], z[:N, :N])
    W[N:, :N] = weight(x, J[1, 0], delta[1, 0], sigma[1, 0], z[N:, :N])
    W[:N, N:] = weight(x, -J[0, 1], delta[0, 1], sigma[0, 1], z[:N, N:])
    W[N:, N:] = weight(x, -J[1, 1], delta[1, 1], sigma[1, 1], z[N:, N:])
    return W


def generate_parameter(N, J, delta, sigma, seed=None):
    """
    Generate 2N-by-2N connectivity matrix and "latent" variable z.
    """
    rs = numpy.random.RandomState(seed)
    z = rs.uniform(size=(2 * N, 2 * N))
    return generate_weight(N, J, delta, sigma, z), z
