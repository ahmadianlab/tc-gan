from __future__ import print_function, division

import scipy.io
import theano.tensor as T
import theano
import numpy

from ..gradient_expressions import make_w_batch


J = T.matrix("j", "float32")
D = T.matrix("d", "float32")
S = T.matrix("s", "float32")
Z = T.tensor3("z", "float32")
X = T.vector("x", "float32")
N = T.scalar("N", "int32")

w_sym = make_w_batch.make_W_with_x(Z, J, D, S, N, X)
w_fun = theano.function([Z, J, D, S, N, X], w_sym, allow_input_downcast=True)


def numeric_w(Z, J, D, S):
    _, n, n2 = Z.shape
    assert n == n2
    N = n // 2
    assert 2 * N == n
    x = numpy.linspace(-.5, .5, N)
    return w_fun(Z, J, D, S, N, x)


def test_weight():
    netmat = scipy.io.loadmat('target_parameters_GAN-SSN_Ne51-Zs.mat')
    scale_mat = 8
    mz = netmat['Zs']
    J = netmat['Targetparams']['Jlow'][0, 0]
    D = netmat['Targetparams']['dJ'][0, 0]
    S = netmat['Targetparams']['sigmas'][0, 0] / scale_mat
    W_desired = netmat['W'].toarray()
    N = mz.shape[0]

    # Make sure that matrix format is as I expected by checking the
    # Dale's law:
    assert W_desired[:, :N].min() >= 0
    assert W_desired[:, N:].max() >= 0

    Z = numpy.zeros((2*N, 2*N))
    Z[:N, :N] = mz[:, :, 0, 0]
    Z[N:, :N] = mz[:, :, 1, 0]
    Z[:N, N:] = mz[:, :, 0, 1]
    Z[N:, N:] = mz[:, :, 1, 1]
    W = numeric_w(numpy.asarray([Z]), J, D, S)
    assert W.ndim == 3
    assert W.shape[0] == 1

    numpy.testing.assert_allclose(W[0], W_desired, atol=1e-6)


if __name__ == '__main__':
    test_weight()
