from __future__ import print_function, division

import scipy.io
import theano.tensor as T
import theano
import numpy
from matplotlib import pyplot

from ..gradient_expressions import make_w_batch
from ..ssnode import solve_dynamics
import stimuli


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
    conn_param = scipy.io.loadmat('target_parameters_GAN-SSN_Ne51-Zs.mat')
    scale_mat = 8
    mz = conn_param['Zs']
    J = conn_param['Targetparams']['Jlow'][0, 0]
    D = conn_param['Targetparams']['dJ'][0, 0]
    S = conn_param['Targetparams']['sigmas'][0, 0] / scale_mat
    W_desired = conn_param['W'].toarray()
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


def asserting_div(num, divisor, desired_reminder):
    got, reminder = divmod(num, divisor)
    assert reminder == desired_reminder
    return got


def test_tuning_curve():
    conn_param = scipy.io.loadmat('target_parameters_GAN-SSN_Ne51-Zs.mat')
    model_param = scipy.io.loadmat('training_data_TCs_Ne51-Zs.mat')

    W = conn_param['W'].toarray()
    L_mat = model_param['Modelparams'][0, 0]['L'][0, 0]
    bandwidths = model_param['Modelparams'][0, 0]['bandwidths'][0] / L_mat
    smoothness = model_param['Modelparams'][0, 0]['l_margin'][0, 0] / L_mat
    contrast = model_param['Modelparams'][0, 0]['c'][0, 0]
    n_sites = int(model_param['Modelparams'][0, 0]['Ne'][0, 0])
    coe_value = float(model_param['Modelparams'][0, 0]['k'][0, 0])
    exp_value = float(model_param['Modelparams'][0, 0]['n'][0, 0])
    E_Tuning_desired = model_param['E_Tuning']      # shape: (N_data, nb)

    center = asserting_div(n_sites, 2, 1)
    ofs = asserting_div(len(E_Tuning_desired), 2, 1)
    i_beg = center - ofs
    i_end = center + ofs + 1

    X = numpy.linspace(-0.5, 0.5, n_sites)
    BAND_IN = stimuli.input(bandwidths, X, smoothness, contrast)

    fps = [solve_dynamics(None, W, ext, k=coe_value, n=exp_value,
                          r0=numpy.zeros(2 * n_sites))
           for ext in BAND_IN]
    E_Tuning_actual = numpy.array([x[i_beg:i_end] for x in fps]).T

    if False:
        # Plot two curves for debugging:
        pyplot.plot(E_Tuning_desired.T, label='desired')
        pyplot.plot(E_Tuning_actual.T, label='actual')
        pyplot.legend(loc='best')
        pyplot.show()

    numpy.testing.assert_allclose(E_Tuning_actual, E_Tuning_desired, rtol=0.1)


if __name__ == '__main__':
    test_weight()
    test_tuning_curve()
