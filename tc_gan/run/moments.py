"""

Run SSN-Moment Matching

"""

from __future__ import print_function

import theano
import theano.tensor as T
import lasagne
import numpy as np

from .. import utils
from ..gradient_expressions.utils import subsample_neurons, \
    sample_sites_from_stim_space
from ..gradient_expressions import make_w_batch as make_w
from ..gradient_expressions import SS_grad as SSgrad
from .. import ssnode as SSsolve
from .gan import add_learning_options, do_learning

import time

from .. import stimuli


def learn(
        seed, gen_learn_rate,
        loss, n_samples, lam,
        rate_cost, rate_penalty_threshold, rate_penalty_no_I,
        n_sites, IO_type, rate_hard_bound, rate_soft_bound, dt, max_iter,
        true_IO_type, truth_size, truth_seed, bandwidths,
        sample_sites, track_offset_identity, init_disturbance,
        contrast,
        offsets,
        datastore, J0, D0, S0,
        timetest, convtest, testDW, DRtest, truth_size_per_batch, use_inhibitory,
        # ignore:
        gan, driver):
    iterations = driver.iterations
    quiet = driver.quiet

    print(datastore)

    sample_sites = sample_sites_from_stim_space(sample_sites, n_sites)

    print(sample_sites)

    def make_functions(**kwds):
        return make_MOMENT_functions(lam=lam, **kwds)

    train_update = MOMENT_update

    rate_cost = float(rate_cost)
    np.random.seed(seed)

    smoothness = 0.03125

    coe_value = .01 #k
    exp_value = 2.2 #n

    ssn_params = dict(
        dt=dt,
        max_iter=max_iter,
        rate_soft_bound=rate_soft_bound,
        rate_hard_bound=rate_hard_bound,
        io_type=IO_type,
        k=coe_value,
        n=exp_value,
        r0=np.zeros(2 * n_sites),
    )

    #specifying the shape of model/input
    n = theano.shared(n_sites,name = "n_sites")
    nz = theano.shared(n_samples,name = 'n_samples')
    nb = theano.shared(len(bandwidths)*len(contrast),name = 'n_stim')

    ##getting regular nums##
    N = int(n.get_value())
    NZ = int(nz.get_value())
    NB = int(nb.get_value())
    ###

    #array that computes the positions
    X = theano.shared(np.linspace(-.5,.5,n.get_value()).astype("float32"),name = "positions")
    
    pos_num = X.get_value()

    BAND_IN = stimuli.input(bandwidths, X.get_value(), smoothness, contrast)
    
    data = []
        
    if not true_IO_type:
        true_IO_type = IO_type
    print("Generating the truth...")

    data, _ = SSsolve.sample_tuning_curves(
        sample_sites=sample_sites,
        NZ=truth_size,
        seed=truth_seed,
        bandwidths=bandwidths,
        smoothness=smoothness,
        contrast = contrast,
        offset = offsets,
        N=n_sites,
        rate_stop_at=200,
        track_offset_identity=track_offset_identity,
        include_inhibitory_neurons = use_inhibitory,
        **dict(ssn_params, io_type=true_IO_type))
    
    print("DONE")
    data = data.T      # shape: (ncondition,N_data, nb)
    print(data.shape)
        
    # Check for sanity:
    n_stim = len(bandwidths) * len(contrast)  # number of stimulus conditions

    if track_offset_identity:
        assert n_stim * len(sample_sites) * (2 if use_inhibitory else 1) == data.shape[-1]
    else:
        assert n_stim == data.shape[-1]

    #defining all the parameters that we might want to train

    print(n_sites)

    exp = theano.shared(exp_value,name = "exp")
    coe = theano.shared(coe_value,name = "coe")

    #these are parameters we will use to test the GAN
    J2 = theano.shared(np.log(np.array(J0)).astype("float32"), name="j")
    D2 = theano.shared(np.log(np.array(D0)).astype("float32"), name="d")
    S2 = theano.shared(np.log(np.array(S0)).astype("float32"), name="s")

    Jp2 = T.exp(J2)
    Dp2 = T.exp(D2)
    Sp2 = T.exp(S2)

    #these are the parameters to be fit
    dp = init_disturbance
    
    if isinstance(dp, tuple):
        J_dis, D_dis, S_dis = dp
    else:
        J_dis = D_dis = S_dis = dp
        
    J_dis = np.array(J_dis) * np.ones((2, 2))
    D_dis = np.array(D_dis) * np.ones((2, 2))
    S_dis = np.array(S_dis) * np.ones((2, 2))

    J = theano.shared(J2.get_value() + J_dis, name = "j")
    D = theano.shared(D2.get_value() + D_dis, name = "d")
    S = theano.shared(S2.get_value() + S_dis, name = "s")

#    J = theano.shared(np.log(np.array([[.01,.01],[.02,.01]])).astype("float64"),name = "j")
#    D = theano.shared(np.log(np.array([[.2,.2],[.3,.2]])).astype("float64"),name = "d")
#    S = theano.shared(np.log(np.array([[.1,.1],[.1,.1]])).astype("float64"),name = "s")

    Jp = T.exp(J)
    Dp = T.exp(D)
    Sp = T.exp(S)

    #compute jacobian of the primed variables w.r.t. J,D,S.
    dJpJ = T.reshape(T.jacobian(T.reshape(Jp,[-1]),J),[2,2,2,2])
    dDpD = T.reshape(T.jacobian(T.reshape(Dp,[-1]),D),[2,2,2,2])
    dSpS = T.reshape(T.jacobian(T.reshape(Sp,[-1]),S),[2,2,2,2])
    
    print(BAND_IN.shape)

    #theano variable for the random samples
    Z = T.tensor3("z","float32")

    #symbolic W
    ww = make_w.make_W_with_x(Z,Jp,Dp,Sp,n,X)

    ww2 = make_w.make_W_with_x(Z,Jp2,Dp2,Sp2,n,X)

    #the next 3 are of shape [nz,2N,2N,2,2]
    dwdj = T.tile(make_w.make_WJ_with_x(Z,Jp,Dp,Sp,n,X,dJpJ),(NZ,1,1,1,1))#deriv. of W w.r.t. J
    dwdd = make_w.make_WD_with_x(Z,Jp,Dp,Sp,n,X,dDpD)#deriv. of W w.r.t. D
    dwds = make_w.make_WS_with_x(Z,Jp,Dp,Sp,n,X,dSpS)#deriv of W w.r.t. S

    DWj_slow = T.reshape(T.jacobian(T.reshape(ww,[-1]),J),[nz,2*n,2*n,2,2])
    DWd_slow = T.reshape(T.jacobian(T.reshape(ww,[-1]),D),[nz,2*n,2*n,2,2])
    DWs_slow = T.reshape(T.jacobian(T.reshape(ww,[-1]),S),[nz,2*n,2*n,2,2])

    #function to get W given Z
    W = theano.function([Z],ww,allow_input_downcast = True,on_unused_input = "ignore")
    W_test = theano.function([Z],ww2,allow_input_downcast = True,on_unused_input = "ignore")

    #get deriv. of W given Z
    DWj = theano.function([Z],dwdj,allow_input_downcast = True,on_unused_input = "ignore")
    DWd = theano.function([Z],dwdd,allow_input_downcast = True,on_unused_input = "ignore")
    DWs = theano.function([Z],dwds,allow_input_downcast = True,on_unused_input = "ignore")

    DWj_slow_f = theano.function([Z],DWj_slow,allow_input_downcast = True,on_unused_input = "ignore")
    DWd_slow_f = theano.function([Z],DWd_slow,allow_input_downcast = True,on_unused_input = "ignore")
    DWs_slow_f = theano.function([Z],DWs_slow,allow_input_downcast = True,on_unused_input = "ignore")

    #a random Z sample for use in testing
    Ztest = np.random.rand(NZ,2*N,2*N).astype("float32")
    #now we need to get a function to generate dr/dth from dw/dth

    #variables for rates and inputs
    rvec = T.tensor3("rvec","float32")
    ivec = T.matrix("ivec","float32")

    #DrDth tensor expressions
    WRgrad_params = dict(
        R=rvec, W=ww, I=ivec,
        n=exp, k=coe, nz=NZ, nb=NB, N=N,
        io_type=IO_type,
        r0=rate_soft_bound,
        r1=rate_hard_bound,
        CGAN = True,
    )
    
    dRdJ_exp = SSgrad.WRgrad_batch(DW=dwdj, **WRgrad_params)
    dRdD_exp = SSgrad.WRgrad_batch(DW=dwdd, **WRgrad_params)
    dRdS_exp = SSgrad.WRgrad_batch(DW=dwds, **WRgrad_params)

    dRdJ = theano.function([rvec,ivec,Z],dRdJ_exp,allow_input_downcast = True)
    dRdD = theano.function([rvec,ivec,Z],dRdD_exp,allow_input_downcast = True)
    dRdS = theano.function([rvec,ivec,Z],dRdS_exp,allow_input_downcast = True)
    
    G_train_func,G_loss_func,get_reduced = make_functions(
        rate_vector=rvec, NZ=NZ, NB=NB, LOSS=loss,
        sample_sites=sample_sites, track_offset_identity=track_offset_identity,use_inhibitory = use_inhibitory,
        g_lr=gen_learn_rate, rate_cost=rate_cost,
        rate_penalty_threshold=rate_penalty_threshold,
        rate_penalty_no_I=rate_penalty_no_I,
        ivec=ivec, Z=Z, J=J, D=D, S=S, N=N,
        R_grad=[dRdJ_exp, dRdD_exp, dRdS_exp])

    #Now we set up values to use in testing.

    inp = BAND_IN

    def saverow_learning(row):
        datastore.tables.saverow("learning.csv", row, echo=not quiet)
    saverow_learning("epoch,Gloss,Dloss,Daccuracy,SSsolve_time,gradient_time,model_convergence,model_unused")

    for k in range(iterations):
        Gloss,rtest,true,model_info,SSsolve_time,gradient_time = train_update(G_train_func,iterations,N,NZ,NB,data,W,W_test, BAND_IN,ssn_params,get_reduced,J,D,S,truth_size_per_batch)

        saverow_learning(
            [k, Gloss,
             SSsolve_time,
             gradient_time,
             model_info.rejections,
             model_info.unused])

        GZmean = get_reduced(rtest).mean(axis = 0)
        Dmean = true.mean(axis = 0)

        GZvar = get_reduced(rtest).var(axis = 0)
        Dvar = true.var(axis = 0)

        datastore.tables.saverow('TC_mean.csv', list(GZmean) + list(Dmean) + list(GZvar) + list(Dvar))

        jj = J.get_value()
        dd = D.get_value()
        ss = S.get_value()
        
        allpar = np.reshape(np.concatenate([jj,dd,ss]),[-1]).tolist()
        datastore.tables.saverow('generator.csv', [k] + allpar)

        if k == 3000:
            break

###I need theano functions that compute each of the things we want to moment match.

def suppresion_index(TC):
    #TC is a tensor with shape [Nbatch, nbandwidth]
    return 1. - (TC[:,-1]/T.max(TC,axis = 1,keepdims = True))
             
def max_rate(TC):
    return T.max(TC,axis = 1)

def peak_width(TC):
    norm = T.sum(TC,axis = 1,keepdims = True)
    return 1./T.sum((TC/norm)**2,axis = 1)
                    
def MOMENT_update(G_train_func,iterations,N,NZ,NB,data,W,W_test,INPUT,ssn_params,get_reduced,J,D,S,truth_size_per_batch,WG_repeat = 5):

    '''
    Conditional WGAN update function:
    args:
     D_train_func - function that updates disc. parameters
     G_train_func - function tha tupdates generator parameters
     iterations - number of update steps to perform
     N - number of sites in SSN network
     NZ - number of Zs to sample per batch
     NB - number of bandwidths
     data - list of datasets for each condition
     data_cond - list of conditions, indices must match the data sets in the data list
     W - function to generate weight matrices from Z samples
     W_test - 
     inp_function - a function that takes conditions and generates the corresponding network inputs.
     ssn_params - list of SSN parameter shared variable.
     D_acc - function to compute discriminator accuracy
     get_reduced - function to get reduced set of sites.
     J - J variable
     D - D variable
     S - sigma variable
     truth_size_per_batch - number of true samples to take for each condition per batch
     WG_repeat - wgan ncritic parameter

    '''
    
    SSsolve_time = utils.StopWatch()
    gradient_time = utils.StopWatch()

    def Z_W_gen():
        while True:
            ztest = np.random.rand(1, 2*N, 2*N)
            wtest, = W(ztest)
            yield ztest[0], wtest

    # Generate "fake" samples from the fitted model.  Since the
    # generator does not change in the loop over updates of the
    # discriminator, we generate the whole samples at once.  This
    # gives us 40% speedup in asym_tanh case:

    #the data
    idx = np.random.choice(len(data), truth_size_per_batch)
    true = data[idx]

    #I need to generate the inputs for these conditions
    
    ####
    
    #generated samples
    #
    # Update discriminator/critic given NZ true and fake samples:


    with SSsolve_time:
        ztest,rtest,model_info = SSsolve.find_fixed_points(NZ,Z_W_gen(),INPUT,**ssn_params)

    with gradient_time:
        Gloss = G_train_func(rtest,true.mean(axis = 0),true.var(axis = 0),INPUT,ztest)

    return Gloss,rtest,true,model_info,SSsolve_time.sum(),gradient_time.sum()

def make_MOMENT_functions(rate_vector,sample_sites,NZ,NB,LOSS,g_lr,rate_cost,rate_penalty_threshold,rate_penalty_no_I,ivec,Z,J,D,S,N,R_grad,track_offset_identity,lam,use_inhibitory):

    print(sample_sites)
    
    # Convert rate_vector of shape [NZ, NB, 2N] to an array of shape INSHAPE
    red_R_fake = subsample_neurons(rate_vector, N=N, NZ=NZ, NB=NB,
                                   sample_sites=sample_sites,
                                   include_inhibitory_neurons = use_inhibitory,
                                   track_offset_identity=track_offset_identity)

    get_reduced = theano.function([rate_vector],red_R_fake,allow_input_downcast = True)

    MEAN = red_R_fake.mean(axis = 0)
    VARI = red_R_fake.var(axis = 0)

    dat_mean = T.vector("mean","float32")
    dat_vari = T.vector("variance","float32")

    r0 = MEAN.mean()
    
    loss_exp = (((dat_mean - MEAN)/r0)**2).sum() + lam * (((dat_vari - VARI)/(r0**2))**2).sum()
    
    if rate_penalty_no_I:
        penalized_rate = rate_vector[:, :, :N]
    else:
        penalized_rate = rate_vector
        
    loss_exp_train = loss_exp + rate_cost * SSgrad.rectify(penalized_rate - rate_penalty_threshold).mean()

    #make loss functions
    loss = theano.function([rate_vector,dat_mean,dat_vari],loss_exp,allow_input_downcast = True)
    
    #Computing the G gradient!
    #
    #We have to do this by hand because hte SS soluytion is not written in a way that theano can solve
    #
    #to get the grads w.r.t. the generators parameters we need to do a jacobian 
    fake_dis_grad = T.jacobian(loss_exp_train,rate_vector) #gradient of generator loss w.r.t rates [30,45,402]
    
    nstim = NB
        
#    fake_dis_grad = T.reshape(fake_dis_grad,[NZ,nstim,2*N])

    #reshape the generator gradient to fit with Dr/Dth
    fake_dis_grad_expanded = fake_dis_grad.dimshuffle([0,1,2,'x','x'])

    #putthem together and sum of the z,b,and N axes to get a [2,2] tensor that is the gradient of the loss w.r.t. parameters
    dLdJ_exp = (fake_dis_grad_expanded*R_grad[0]).sum(axis = (0,1,2))
    dLdD_exp = (fake_dis_grad_expanded*R_grad[1]).sum(axis = (0,1,2))
    dLdS_exp = (fake_dis_grad_expanded*R_grad[2]).sum(axis = (0,1,2))

    dLdJ = theano.function([rate_vector,dat_mean,dat_vari,ivec,Z],dLdJ_exp,allow_input_downcast = True)
    dLdD = theano.function([rate_vector,dat_mean,dat_vari,ivec,Z],dLdD_exp,allow_input_downcast = True)
    dLdS = theano.function([rate_vector,dat_mean,dat_vari,ivec,Z],dLdS_exp,allow_input_downcast = True)
    ####

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    b1 = .5
    b2 = .9

    cut_value = 10.
    
    def cut(tensor):
        return lasagne.updates.norm_constraint(tensor,cut_value)
    
    G_updates = lasagne.updates.sgd([cut(dLdJ_exp),cut(dLdD_exp),cut(dLdS_exp)],[J,D,S], g_lr)#,beta1 = b1,beta2 = b2)

    G_train_func = theano.function([rate_vector,dat_mean,dat_vari,ivec,Z],loss_exp_train,updates = G_updates,allow_input_downcast = True)
    
    G_loss_func = theano.function([rate_vector,dat_mean,dat_vari,ivec,Z],loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')

    return G_train_func, G_loss_func, get_reduced

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for random numbers (default: %(default)s)')
    parser.add_argument(
        '--init-disturbance', default=-1.5, type=eval,
        help='''Initial disturbance to the parameter.  If it is
        evaluated to be a 3-tuple, the components are used for the
        disturbance for J, D (delta), S (sigma), respectively.  It
        accepts any Python expression.
        (default: %(default)s)''')
    parser.add_argument(
        '--gen-learn-rate', default=0.001, type=float,
        help='Learning rate for generator (default: %(default)s)')
    # Cannot use MATLAB data at the moment:
    """
    parser.add_argument(
        '--use-data', default=False, action='store_true',
        help='Use data (True) or generate our own TC samples (False) (default: %(default)s)')
    """
    parser.add_argument(
        '--N', '-N', dest='n_sites', default=201, type=int,
        help='''Number of excitatory neurons in SSN. (default:
        %(default)s)''')
    parser.add_argument(
        '--dt', default=5e-4, type=float,
        help='''Time step used for SSN fixed point finder.
        (default: %(default)s)''')
    parser.add_argument(
        '--max_iter', default=100000, type=int,
        help='''Number of time steps used for SSN fixed point finder.
        (default: %(default)s)''')
    parser.add_argument(
        '--sample-sites', default=[0,.25,.5], type=utils.csv_line(float),
        help='''Locations (offsets) of neurons to be sampled from SSN in the
        "bandwidth" space [-1, 1].  0 means the center of the
        network. (default: %(default)s)''')
    parser.add_argument(
        '--track_offset_identity', action='store_true',
        help='''If False (default), squash all neurons into NZ axis;
        i.e., forget from which probe offset the neurons are sampled.
        If True, stack samples into NB axis; i.e., let discriminator
        know that those neurons are from the different offset of the
        same SSN.''')

    parser.add_argument(
        '--use_inhibitory', action='store_true',
        help='''If False (default) use only excitatory TC. If True use excitatory and inhibitory''')
    parser.add_argument(
        '--contrast',
        default=[5,20],
        type=utils.csv_line(float),
        help='Comma separated value of floats')
    parser.add_argument(
        '--offsets',
        default=[0],
        type=utils.csv_line(float),
        help='Comma separated value of floats')
    parser.add_argument(
        '--IO_type', default="asym_tanh",
        help='Type of nonlinearity to use. Regular ("asym_power"). Linear ("asym_linear"). Tanh ("asym_tanh") (default: %(default)s)')
    parser.add_argument(
        '--true_IO_type', default="asym_power",
        help='''Same as --IO_type but for training data generation.
        --IO_type is used if this option is not given or an empty
        string is passed. (default: %(default)s)''')
    parser.add_argument(
        '--truth_size', default=1000, type=int,
        help='''Number of SSNs to be used to generate ground truth
        data (default: %(default)s)''')
    parser.add_argument(
        '--truth_size_per_batch', default=100, type=int,
        help='''Number of data samples to use for each mean/variance (default: %(default)s)''')
    parser.add_argument(
        '--truth_seed', default=42, type=int,
        help='Seed for generating ground truth data (default: %(default)s)')
    parser.add_argument(
        '--loss', default="CE",
        help='Type of loss to use. Cross-Entropy ("CE") or LSGAN ("LS"). (default: %(default)s)')
    parser.add_argument(
        '--n_samples', default=30, type=eval,
        help='Number of samples to draw from G each step (default: %(default)s)')
    parser.add_argument(
        '--rate_cost', default='0',
        help='The cost of having the rate be large (default: %(default)s)')
    parser.add_argument(
        '--rate_penalty_threshold', default=150.0, type=float,
        help='''The point at which the rate penalty cost kicks in
        (default: %(default)s)''')
    parser.add_argument(
        '--rate_penalty_no_I', action='store_true', default=False,
        help='''If specified, do not penalize large inhibitory rate.
        (default: %(default)s)''')
    parser.add_argument(
        '--rate_soft_bound', default=200, type=float,
        help='rate_soft_bound=r0 (default: %(default)s)')
    parser.add_argument(
        '--rate_hard_bound', default=1000, type=float,
        help='rate_hard_bound=r1 (default: %(default)s)')
    parser.add_argument(
        '--lam', default=.1, type=float,
        help='The complexity penalty for the D (default: %(default)s)')


    parser.add_argument(
        '--timetest',default=False, action='store_true',
        help='perform a timing test on the SS solver')
    parser.add_argument(
        '--convtest',default=False, action='store_true',
        help='perform a convergence test on the SS solver')
    parser.add_argument(
        '--testDW',default=False, action='store_true',
        help='Test the W gradient')
    parser.add_argument(
        '--DRtest',default=False, action='store_true',
        help='test the R gradient')

    add_learning_options(parser)
    parser.set_defaults(
        datastore_template='logfiles/moments_{IO_type}_{rate_cost}_{lam}_{n_samples}_{init_disturbance}',
    )

    ns = parser.parse_args(args)
    do_learning(learn, vars(ns), script_file=__file__)


if __name__ == "__main__":
    main()
