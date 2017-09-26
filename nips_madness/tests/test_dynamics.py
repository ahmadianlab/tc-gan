from __future__ import print_function, division

import scipy.io
import theano.tensor as T
import theano
import numpy
from matplotlib import pyplot
import numpy as np

from ..gradient_expressions import make_w_batch
from ..ssnode import solve_dynamics, fixed_point, find_fixed_points
import stimuli

from ..gradient_expressions import make_w_batch as make_w
from ..gradient_expressions import SS_grad as SSgrad


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


def test_tuning_curve_asym_linear(io_type='asym_linear', method='parallel'):
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
    BAND_IN = stimuli.input(bandwidths, X, smoothness, [contrast])

    dummy_z = object()
    (actual_z,), (fps,), info = find_fixed_points(
        1, iter([(dummy_z, W)]), BAND_IN,
        # Model parameters:
        k=coe_value, n=exp_value,
        r0=numpy.zeros(2 * n_sites),
        io_type=io_type,
        # Solver parameters:
        method=method,
        check=True,
    )
    E_Tuning_actual = numpy.array([x[i_beg:i_end] for x in fps]).T

    if False:
        # Plot two curves for debugging:
        pyplot.plot(E_Tuning_desired.T, label='desired')
        pyplot.plot(E_Tuning_actual.T, label='actual')
        pyplot.legend(loc='best')
        pyplot.show()

    numpy.testing.assert_allclose(E_Tuning_actual, E_Tuning_desired, rtol=0.1)


def test_tuning_curve_asym_power():
    test_tuning_curve_asym_linear(io_type='asym_power')


def test_tuning_curve_asym_tanh():
    test_tuning_curve_asym_linear(io_type='asym_tanh')


def test_inf():
    sol = fixed_point(W=[[2, 0], [0, 0]], ext=[10, 10],
                      k=1, n=1, r0=[0, 0],
                      max_iter=10000000,
                      io_type='asym_linear')
    assert sol.message == "Reached to rate_stop_at"
    # assert sol.x[0] == numpy.inf
    # TODO: Check it's fine to ignore this.  I think this is fine,
    # since r1 can (or is more likely to) become inf instead of r0.


def test_gradients():
    conn_param = scipy.io.loadmat('target_parameters_GAN-SSN_Ne51-Zs.mat')
    model_param = scipy.io.loadmat('training_data_TCs_Ne51-Zs.mat')


    dub = conn_param['W'].toarray()
    L_mat = model_param['Modelparams'][0, 0]['L'][0, 0]
    bandwidths = model_param['Modelparams'][0, 0]['bandwidths'][0] / L_mat
    smoothness = model_param['Modelparams'][0, 0]['l_margin'][0, 0] / L_mat
    contrast = model_param['Modelparams'][0, 0]['c'][0, 0]
    n_sites = int(model_param['Modelparams'][0, 0]['Ne'][0, 0])
    coe_value = float(model_param['Modelparams'][0, 0]['k'][0, 0])
    exp_value = float(model_param['Modelparams'][0, 0]['n'][0, 0])
    E_Tuning_desired = model_param['E_Tuning']      # shape: (N_data, nb)

    scale_mat = 8
    mz = conn_param['Zs']
    J = conn_param['Targetparams']['Jlow'][0, 0]
    D = conn_param['Targetparams']['dJ'][0, 0]
    S = conn_param['Targetparams']['sigmas'][0, 0] / scale_mat

    center = asserting_div(n_sites, 2, 1)
    ofs = asserting_div(len(E_Tuning_desired), 2, 1)
    i_beg = center - ofs
    i_end = center + ofs + 1

    X = numpy.linspace(-0.5, 0.5, n_sites)
    BAND_IN = stimuli.input(bandwidths, X, smoothness, [contrast])

    ZZ = np.reshape(conn_param['Zs'].transpose([2,0,3,1]),(1,2*n_sites,2*n_sites))
    print(ZZ.shape)
    
    NW = numeric_w(ZZ, J, D, S)
    print(NW.shape)
    
    print(np.abs((NW - dub)).mean())
    

    ##GRADIENTS

    exp = theano.shared(exp_value,name = "exp")
    coe = theano.shared(coe_value,name = "coe")

    #these are parameters we will use to test the GAN
    J = theano.shared(np.array(J).astype("float64"),name = "j")
    D = theano.shared(np.array(D).astype("float64"),name = "d")
    S = theano.shared(np.array(S).astype("float64"),name = "s")

#    J = theano.shared(np.log(np.array(J)).astype("float64"),name = "j")
#    D = theano.shared(np.log(np.array(D)).astype("float64"),name = "d")
#    S = theano.shared(np.log(np.array(S)).astype("float64"),name = "s")
    
    print(J.get_value())
    print(D.get_value())
    print(S.get_value())

    Jp = J
    Dp = D
    Sp = S

#    Jp = T.exp(J)
#    Dp = T.exp(D)
#    Sp = T.exp(S)


    #compute jacobian of the primed variables w.r.t. J,D,S.
    dJpJ = T.reshape(T.jacobian(T.reshape(Jp,[-1]),J),[2,2,2,2])
    dDpD = T.reshape(T.jacobian(T.reshape(Dp,[-1]),D),[2,2,2,2])
    dSpS = T.reshape(T.jacobian(T.reshape(Sp,[-1]),S),[2,2,2,2])

    print(dJpJ.eval())

    #specifying the shape of model/input
    n = theano.shared(n_sites,name = "n_sites")
    nz = theano.shared(1,name = 'n_samples')
    nb = theano.shared(E_Tuning_desired.shape[1],name = 'n_stim')

    ##getting regular nums##
    N = int(n.get_value())
    NZ = int(nz.get_value())
    NB = int(nb.get_value())
    m = 1
    ###

    #theano variable for the random samples
    Z = T.tensor3("z","float32")

    #symbolic W
    ww = make_w.make_W_with_x(Z,Jp,Dp,Sp,n,X)

    #the next 3 are of shape [nz,2N,2N,2,2]
    dwdj = T.tile(make_w.make_WJ_with_x(Z,Jp,Dp,Sp,n,X,dJpJ),(NZ,1,1,1,1))#deriv. of W w.r.t. J
    dwdd = make_w.make_WD_with_x(Z,Jp,Dp,Sp,n,X,dDpD)#deriv. of W w.r.t. D
    dwds = make_w.make_WS_with_x(Z,Jp,Dp,Sp,n,X,dSpS)#deriv of W w.r.t. S

    #function to get W given Z
    W = theano.function([Z],ww,allow_input_downcast = True,on_unused_input = "ignore")

    print((W(ZZ) - dub).mean())

    #get deriv. of W given Z
    DWj = theano.function([Z],dwdj,allow_input_downcast = True,on_unused_input = "ignore")
    DWd = theano.function([Z],dwdd,allow_input_downcast = True,on_unused_input = "ignore")
    DWs = theano.function([Z],dwds,allow_input_downcast = True,on_unused_input = "ignore")

    #variables for rates and inputs
    rvec = T.tensor3("rvec","float32")
    ivec = T.matrix("ivec","float32")

    #DrDth tensor expressions
    dRdJ_exp = SSgrad.WRgrad_batch(rvec,ww,dwdj,ivec,exp,coe,NZ,NB,N)
    dRdD_exp = SSgrad.WRgrad_batch(rvec,ww,dwdd,ivec,exp,coe,NZ,NB,N)
    dRdS_exp = SSgrad.WRgrad_batch(rvec,ww,dwds,ivec,exp,coe,NZ,NB,N)

    dRdJ = theano.function([rvec,ivec,Z],dRdJ_exp,allow_input_downcast = True)
    dRdD = theano.function([rvec,ivec,Z],dRdD_exp,allow_input_downcast = True)
    dRdS = theano.function([rvec,ivec,Z],dRdS_exp,allow_input_downcast = True)

    ##DONE COMPUTING GRADIENTS

    print("DW {}".format(DWd(ZZ)[0,0,0,0,0]))

    fps = np.array([[solve_dynamics(W(ZZ)[0], ext, k=coe_value, n=exp_value,
                                    check=True,
                                    r0=numpy.zeros(2 * n_sites))
                     for ext in BAND_IN]])

    DR = dRdD(fps,BAND_IN,ZZ)
    print("DRRRRRR {}".format(DR.mean()))

    print(DR[0,:,24:27,0,0])

    E_Tuning_actual = numpy.array([x[i_beg:i_end] for x in fps[0]]).T

    print(fps.shape)
    print(E_Tuning_actual.shape)

    numpy.testing.assert_allclose(E_Tuning_actual, E_Tuning_desired, rtol=0.1)


if __name__ == '__main__':
    test_weight()
    test_tuning_curve_asym_power()
    test_tuning_curve_asym_linear()
    test_tuning_curve_asym_tanh()
    test_gradients()
