"""
Run SSN-GAN learning.

It stores the learning result in the datastore directory specified by
--datastore or --datastore-template.  Following files are generated in
the datastore:

* learning.csv --- All learning related stats such as
  generator/discriminator losses go into this file.

* disc_learning.csv --- Discriminator specific learning statistics.
  Recorded even during the inner loop for discriminator.

* TC_mean.csv --- Tuning curves averaged over instances (the "z-axis")
  are stored in a row (for each generator step).  The first half is
  the mean of the samples from "fake" SSN and the second half is that
  of the "true" SSN.

* generator.csv --- Generator parameters.  Logarithm of actual values
  are stored.  Three 2x2 matrices J, D and S (sigma) are stored in the
  2nd to 13th columns after they are concatenated and flattened.  The
  first column stores the generator step.  Each row corresponds to
  each generator step.

* disc_param_stats.csv --- Normalized norms (2-norm divided by number
  of elements) of parameters in each layers.

* info.json --- It stores parameters used for each run and some
  environment information such as the Git revision of this repository.

* disc_param/last.npz --- Snapshot of the discriminator parameters.
  Saved for each `disc_param_save_interval` generator updates.
  See: --disc-param-save-interval, --disc-param-template

"""

from __future__ import print_function

from types import SimpleNamespace

import theano
import theano.tensor as T
import lasagne
import numpy as np

from .. import execution
from .. import utils
from ..gradient_expressions.utils import subsample_neurons, \
    sample_sites_from_stim_space
import discriminators.simple_discriminator as SD
from ..gradient_expressions import make_w_batch as make_w
from ..gradient_expressions import SS_grad as SSgrad
from .. import lasagne_param_file
from .. import ssnode as SSsolve

import time

from .. import stimuli


def saveheader_disc_param_stats(datastore, discriminator):
    header = ['gen_step', 'disc_step'] + [
        '{}.nnorm'.format(p.name)  # Normalized NORM
        for p in lasagne.layers.get_all_params(discriminator, trainable=True)
    ]
    datastore.tables.saverow('disc_param_stats.csv', header)


def saverow_disc_param_stats(datastore, discriminator, gen_step, disc_step):
    """
    Save normalized norms of `discriminator`.

    This function also checks for finiteness of the parameter and
    raises an error when found.  "Downloading" discriminator parameter
    is (likely) a time-consuming operation so it makes sense to do it
    here.

    """
    nnorms = [
        np.linalg.norm(arr.flatten()) / arr.size
        for arr in lasagne.layers.get_all_param_values(discriminator)
    ]
    row = [gen_step, disc_step] + nnorms
    datastore.tables.saverow('disc_param_stats.csv', row)

    assert np.isfinite(nnorms).all()


def learn(
        iterations, seed, gen_learn_rate, disc_learn_rate,
        loss, layers, n_samples, WGAN_lambda,
        WGAN_n_critic0,
        rate_cost, rate_penalty_threshold, rate_penalty_no_I,
        n_sites, IO_type, rate_hard_bound, rate_soft_bound, dt, max_iter,
        true_IO_type, truth_size, truth_seed, bandwidths,
        sample_sites, track_offset_identity, init_disturbance, quiet,
        contrast,
        disc_normalization, disc_param_save_interval, disc_param_template,
        disc_param_save_on_error,
        datastore, J0, D0, S0,
        timetest, convtest, testDW, DRtest):

    print(datastore)

    bandwidths = np.array(bandwidths)
    sample_sites = sample_sites_from_stim_space(sample_sites, n_sites)

    WGAN = loss == 'WD'
    if WGAN:
        def make_functions(**kwds):
            return make_WGAN_functions(WGAN_lambda=WGAN_lambda, **kwds)
        train_update = WGAN_update
    else:
        make_functions = make_RGAN_functions
        train_update = RGAN_update

    rate_cost = float(rate_cost)
    np.random.seed(seed)

    smoothness = 0.03125
    coe_value = 0.01  # k
    exp_value = 2.2   # n

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

    if not true_IO_type:
        true_IO_type = IO_type
    print("Generating the truth...")
    data, _ = SSsolve.sample_tuning_curves(
        sample_sites=sample_sites,
        NZ=truth_size,
        seed=truth_seed,
        bandwidths=bandwidths,
        smoothness=smoothness,
        contrast=contrast,
        N=n_sites,
        track_offset_identity=track_offset_identity,
        **dict(ssn_params, io_type=true_IO_type))
    print("DONE")
    data = np.array(data.T)      # shape: (N_data, nb)

    # Check for sanity:
    n_stim = len(bandwidths) * len(contrast)  # number of stimulus conditions
    if track_offset_identity:
        assert n_stim * len(sample_sites) == data.shape[-1]
    else:
        assert n_stim == data.shape[-1]

    #defining all the parameters that we might want to train

    print(n_sites)

    exp = theano.shared(exp_value,name = "exp")
    coe = theano.shared(coe_value,name = "coe")

    #these are parameters we will use to test the GAN
    J2 = theano.shared(np.log(np.array(J0)).astype("float64"), name="j")
    D2 = theano.shared(np.log(np.array(D0)).astype("float64"), name="d")
    S2 = theano.shared(np.log(np.array(S0)).astype("float64"), name="s")

    Jp2 = T.exp(J2)
    Dp2 = T.exp(D2)
    Sp2 = T.exp(S2)

    #these are the parammeters to be fit
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

    #specifying the shape of model/input
    n = theano.shared(n_sites,name = "n_sites")
    nz = theano.shared(n_samples,name = 'n_samples')
    nb = theano.shared(n_stim, name='n_stim')

    #array that computes the positions
    X = theano.shared(np.linspace(-.5,.5,n.get_value()).astype("float32"),name = "positions")

    ##getting regular nums##
    N = int(n.get_value())
    NZ = int(nz.get_value())
    NB = int(nb.get_value())
    ###

    BAND_IN = stimuli.input(bandwidths, X.get_value(), smoothness, contrast)
    
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
    )
    dRdJ_exp = SSgrad.WRgrad_batch(DW=dwdj, **WRgrad_params)
    dRdD_exp = SSgrad.WRgrad_batch(DW=dwdd, **WRgrad_params)
    dRdS_exp = SSgrad.WRgrad_batch(DW=dwds, **WRgrad_params)

    dRdJ = theano.function([rvec,ivec,Z],dRdJ_exp,allow_input_downcast = True)
    dRdD = theano.function([rvec,ivec,Z],dRdD_exp,allow_input_downcast = True)
    dRdS = theano.function([rvec,ivec,Z],dRdS_exp,allow_input_downcast = True)

    if convtest:
        zt = np.random.rand(1,2*N,2*N)
        wt = W_test(zt)[0]

        for c in [1.,2.,4.,8.,16.]:
            r  = SSsolve.fixed_point(wt,c*BAND_IN[-1], *ssn_params)
        
            print(np.max(r.x))
        exit()

    if timetest:
        times = []
        EE = 0
        for k in range(100):
            zt = np.random.rand(1,2*N,2*N)
            wt = W_test(zt)[0]

            TT1 = time.time()
            r  = SSsolve.fixed_point(wt,BAND_IN[-1], **ssn_params)

            total = time.time() - TT1
            print(total,r.success)
            times.append(total)


        print("Errors {}".format(np.sum(EE)))
        print("MAX {}".format(np.max(times)))
        print("MIN {}".format(np.min(times)))
        print("MEAN {}".format(np.mean(times)))
        print("MEDIAN {}".format(np.median(times)))

        exit()

    #run gradient descent on W (minimize W*W)
    if testDW:        
        testz = Ztest
        testI = np.random.normal(0,10,(NB,2*N)).astype("float32")
        
        wtest = W(testz)
        ssR = np.asarray([[SSsolve.solve_dynamics(wtest[z],testI[b], *ssn_params)[0].astype("float32") for b in range(len(testI))] for z in range(len(testz))])

        print(ssR.mean())
        print(wtest.mean())
        print(DWj(testz).mean())
                
        print("Start DWtest")
        for k in range(1000):
            wtest = np.reshape(W(testz),[NZ,2*N,2*N,1,1])
            
            dd = DWj(testz)#[nz,2N,2N,2,2]
            dj = 2*(wtest*dd).mean(axis = (0,1,2))

            dd = DWd(testz)#[nz,2N,2N,2,2]
            dD = 2*(wtest*dd).mean(axis = (0,1,2))

            dd = DWs(testz)#[nz,2N,2N,2,2]
            ds = 2*(wtest*dd).mean(axis = (0,1,2))
            
            jnew = J.get_value() - .1*dj
            dnew = D.get_value() - .1*dD
            snew = S.get_value() - .1*ds
            
            J.set_value(jnew.astype("float32"))
            D.set_value(dnew.astype("float32"))
            S.set_value(snew.astype("float32"))
            
            if k % 10 == 0:
                print((wtest*wtest).mean())

        exit()
        #running this will verify that the W gradient is working (by adjusting parameters to minimize the mean sqaured W entry)

    #do gradient descent on R
    if DRtest:
        print("starting DR")
    #I want to test this by adjusting the parameters to give some specified output
         
        inp = BAND_IN
        Ztest = np.random.rand(NZ,2*N,2*N)
        
        WW = W(Ztest)
               
        rr1 = np.array([[SSsolve.solve_dynamics(z,b, **ssn_params)[0] for b in inp] for z in WW])

        DR =[dRdJ(rr1,inp,Ztest),dRdD(rr1,inp,Ztest),dRdS(rr1,inp,Ztest)]
               
        dd = 0.00001

        J.set_value(J.get_value() + np.array([[dd,0],[0,0]]))
        #D.set_value(D.get_value() + np.array([[dd,0],[0,0]]))
        #S.set_value(S.get_value() + np.array([[dd,0.],[0.,0.]]))

        WW = W(Ztest)

        rr2 = np.array([[SSsolve.solve_dynamics(WW[z],inp[b], **dict(ssn_params, r0=rr1[z,b]))[0] for b in range(len(inp))] for z in range(len(WW))])

        print(rr2.max())
        print(rr2.min())

        print(rr1.max())
        print(rr1.min())

        DRa = (rr2 - rr1)/dd
        DRp = DR[0][:,:,:,0,0]

        print(np.max(np.abs(DRa-DRp)))
 
        exit()

    G_train_func,G_loss_func,D_train_func,D_loss_func,D_acc,get_reduced,discriminator = make_functions(
        rate_vector=rvec, NZ=NZ, NB=NB, LOSS=loss, LAYERS=layers,
        sample_sites=sample_sites, track_offset_identity=track_offset_identity,
        d_lr=disc_learn_rate, g_lr=gen_learn_rate, rate_cost=rate_cost,
        rate_penalty_threshold=rate_penalty_threshold,
        rate_penalty_no_I=rate_penalty_no_I,
        ivec=ivec, Z=Z, J=J, D=D, S=S, N=N,
        disc_normalization=disc_normalization,
        R_grad=[dRdJ_exp, dRdD_exp, dRdS_exp])

    #Now we set up values to use in testing.

    inp = BAND_IN

    def saverow_learning(row):
        datastore.tables.saverow("learning.csv", row, echo=not quiet)
    saverow_learning("epoch,Gloss,Dloss,Daccuracy,SSsolve_time,gradient_time,model_convergence,model_unused")

    saveheader_disc_param_stats(datastore, discriminator)
    datastore.tables.saverow('disc_learning.csv', [
        'gen_step', 'disc_step', 'Dloss', 'Daccuracy',
        'SSsolve_time', 'gradient_time',
    ])

    if track_offset_identity:
        truth_size_per_batch = NZ
    else:
        truth_size_per_batch = NZ * len(sample_sites)

    if disc_param_save_on_error:
        train_update = lasagne_param_file.wrap_with_save_on_error(
            discriminator, datastore.path('disc_param', 'pre_error.npz'),
        )(train_update)

    for k in range(iterations):

        Dloss,Gloss,rtest,true,model_info,SSsolve_time,gradient_time = train_update(D_train_func,G_train_func,iterations,N,NZ,NB,data,W,W_test,inp,ssn_params,D_acc,get_reduced,discriminator,J,D,S,truth_size_per_batch,WG_repeat = WGAN_n_critic0 if k == 0 else 5,gen_step=k,datastore=datastore)

        saverow_learning(
            [k, Gloss, Dloss, D_acc(rtest, true),
             SSsolve_time,
             gradient_time,
             model_info.rejections,
             model_info.unused])

        GZmean = get_reduced(rtest).mean(axis = 0)
        Dmean = true.mean(axis = 0)

        datastore.tables.saverow('TC_mean.csv', list(GZmean) + list(Dmean))

        jj = J.get_value()
        dd = D.get_value()
        ss = S.get_value()
        
        allpar = np.reshape(np.concatenate([jj,dd,ss]),[-1]).tolist()
        datastore.tables.saverow('generator.csv', [k] + allpar)

        if disc_param_save_interval > 0 and k % disc_param_save_interval == 0:
            lasagne_param_file.dump(
                discriminator,
                datastore.path('disc_param', disc_param_template.format(k)))


def WGAN_update(D_train_func,G_train_func,iterations,N,NZ,NB,data,W,W_test,inp,ssn_params,D_acc,get_reduced,discriminator,J,D,S,truth_size_per_batch,WG_repeat,gen_step,datastore):

    SSsolve_time = utils.StopWatch()
    gradient_time = utils.StopWatch()

    def Z_W_gen():
        while True:
            ztest = np.random.rand(1, 2*N, 2*N)
            wtest, = W(ztest)
            yield ztest[0], wtest

    model_info = SimpleNamespace(rejections=0, unused=0)

    for rep in range(WG_repeat):
        with SSsolve_time:
            Ztest, rtest, minfo = SSsolve.find_fixed_points(
                NZ, Z_W_gen(), inp,
                **ssn_params)

        model_info.rejections += minfo.rejections
        model_info.unused += minfo.unused

        #the data
        idx = np.random.choice(len(data), truth_size_per_batch)
        true = data[idx]
        ####

        #generated samples
        #
        # Update discriminator/critic given NZ true and fake samples:

        eps = np.random.rand(truth_size_per_batch, 1)

        with gradient_time:
            Dloss = D_train_func(rtest,true,eps*true + (1. - eps)*get_reduced(rtest))

        saverow_disc_param_stats(datastore, discriminator, gen_step, rep)
        datastore.tables.saverow('disc_learning.csv', [
            gen_step, rep, Dloss, D_acc(rtest, true),
            SSsolve_time.times[-1], gradient_time.times[-1],
        ])

    #end D loop

    with gradient_time:
        Gloss = G_train_func(rtest,inp,Ztest)

    return Dloss,Gloss,rtest,true,model_info,SSsolve_time.sum(),gradient_time.sum()

def RGAN_update(D_train_func,G_train_func,iterations,N,NZ,NB,data,W,W_test,inp,ssn_params,D_acc,get_reduced,discriminator,J,D,S,truth_size_per_batch,WG_repeat,gen_step,datastore):

    SSsolve_time = utils.StopWatch()
    gradient_time = utils.StopWatch()

    idx = np.random.choice(len(data), truth_size_per_batch)
    true = data[idx]

    def Z_W_gen():
        while True:
            ztest = np.random.rand(1, 2*N, 2*N)
            wtest, = W(ztest)
            yield ztest[0], wtest

    with SSsolve_time:
        Ztest, Ftest, model_info = SSsolve.find_fixed_points(
            NZ, Z_W_gen(), inp,
            **ssn_params)

    Ftest = np.array(Ftest)
    Ztest = np.array(Ztest)
    rtest = np.array([[c for c in TC] for TC in Ftest])

    with gradient_time:
        Dloss = D_train_func(rtest,true)
        Gloss = G_train_func(rtest,inp,Ztest)

    saverow_disc_param_stats(datastore, discriminator, gen_step, 0)
    datastore.tables.saverow('disc_learning.csv', [
        gen_step, 0, Dloss, D_acc(rtest, true),
        SSsolve_time.times[-1], gradient_time.times[-1],
    ])

    return Dloss,Gloss,rtest,true,model_info,SSsolve_time.sum(),gradient_time.sum()

def make_RGAN_functions(rate_vector,sample_sites,NZ,NB,LOSS,LAYERS,d_lr,g_lr,rate_cost,rate_penalty_threshold,rate_penalty_no_I,ivec,Z,J,D,S,N,R_grad,track_offset_identity,disc_normalization):

    #Defines the input shape for the discriminator network
    if track_offset_identity:
        INSHAPE = [NZ, NB * len(sample_sites)]
    else:
        INSHAPE = [NZ * len(sample_sites), NB]

    ###I want to make a network that takes a tensor of shape [2N] and generates dl/dr

    red_R_true = T.matrix("reduced rates","float32")#data

    # Convert rate_vector of shape [NZ, NB, 2N] to an array of shape INSHAPE
    red_R_fake = subsample_neurons(rate_vector, N=N, NZ=NZ, NB=NB,
                                   sample_sites=sample_sites,
                                   track_offset_identity=track_offset_identity)

    get_reduced = theano.function([rate_vector],red_R_fake,allow_input_downcast = True)

    ##Input Variable Definition

    in_fake = T.log(1. + red_R_fake)
    in_true = T.log(1. + red_R_true)

    #I want to make a network that takes red_R and gives a scalar output

    discriminator = SD.make_net(INSHAPE, LOSS, LAYERS,
                                normalization=disc_normalization)

    #get the outputs
    true_dis_out = lasagne.layers.get_output(discriminator, in_true)
    fake_dis_out = lasagne.layers.get_output(discriminator, in_fake)

    D_acc = theano.function([rate_vector,red_R_true],(true_dis_out.sum() + (1 - fake_dis_out).sum())/(2*NZ),allow_input_downcast = True)

    #make the loss functions
    if LOSS == "CE":
        SM = .9
        true_loss_exp = -SM*np.log(true_dis_out).mean() - (1. - SM)*np.log(1. - true_dis_out).mean() - np.log(1. - fake_dis_out).mean()#discriminator loss
        fake_loss_exp = -np.log(fake_dis_out).mean()#generative loss
    elif LOSS == "LS":
        true_loss_exp = ((true_dis_out - 1.)**2).mean() + ((fake_dis_out + 1.)**2).mean()#discriminator loss
        fake_loss_exp = ((fake_dis_out - 1.)**2).mean()#generative loss
    else:
        print("Invalid loss specified")
        exit()

    if rate_penalty_no_I:
        penalized_rate = rate_vector[:, :, :N]
    else:
        penalized_rate = rate_vector
    fake_loss_exp_train = fake_loss_exp + rate_cost * SSgrad.rectify(penalized_rate - rate_penalty_threshold).mean()

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    D_updates = lasagne.updates.adam(true_loss_exp,lasagne.layers.get_all_params(discriminator, trainable=True), d_lr)#discriminator training function

    #make loss functions
    true_loss = theano.function([red_R_true,rate_vector],true_loss_exp,allow_input_downcast = True)
    fake_loss = theano.function([rate_vector],fake_loss_exp,allow_input_downcast = True)

    #to get the grads w.r.t. the generators parameters we need to do a jacobian 
    fake_dis_grad = T.jacobian(T.flatten(fake_loss_exp_train),rate_vector) #gradient of generator loss w.r.t rates
    fake_dis_grad = T.reshape(fake_dis_grad,[NZ,NB,2*N])

    #reshape the generator gradient to fit with Dr/Dth
    fake_dis_grad_expanded = T.reshape(fake_dis_grad,[NZ,NB,2*N,1,1])

    #putthem together and sum of the z,b,and N axes to get a [2,2] tensor that is the gradient of the loss w.r.t. parameters
    dLdJ_exp = (fake_dis_grad_expanded*R_grad[0]).sum(axis = (0,1,2))
    dLdD_exp = (fake_dis_grad_expanded*R_grad[1]).sum(axis = (0,1,2))
    dLdS_exp = (fake_dis_grad_expanded*R_grad[2]).sum(axis = (0,1,2))

    dLdJ = theano.function([rate_vector,ivec,Z],dLdJ_exp,allow_input_downcast = True)
    dLdD = theano.function([rate_vector,ivec,Z],dLdD_exp,allow_input_downcast = True)
    dLdS = theano.function([rate_vector,ivec,Z],dLdS_exp,allow_input_downcast = True)

    G_updates = lasagne.updates.adam([dLdJ_exp,dLdD_exp,dLdS_exp],[J,D,S], g_lr)

    G_train_func = theano.function([rate_vector,ivec,Z],fake_loss_exp,updates = G_updates,allow_input_downcast = True)
    D_train_func = theano.function([rate_vector,red_R_true],true_loss_exp,updates = D_updates,allow_input_downcast = True)

    G_loss_func = theano.function([rate_vector,ivec,Z],fake_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')
    D_loss_func = theano.function([rate_vector,red_R_true],true_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')

    return G_train_func,G_loss_func,D_train_func,D_loss_func,D_acc,get_reduced,discriminator

def make_WGAN_functions(rate_vector,sample_sites,NZ,NB,LOSS,LAYERS,d_lr,g_lr,rate_cost,rate_penalty_threshold,rate_penalty_no_I,ivec,Z,J,D,S,N,R_grad,track_offset_identity,WGAN_lambda,disc_normalization):

    #Defines the input shape for the discriminator network
    if track_offset_identity:
        INSHAPE = [NZ, NB * len(sample_sites)]
    else:
        INSHAPE = [NZ * len(sample_sites), NB]

    ###I want to make a network that takes a tensor of shape [2N] and generates dl/dr

    red_R_true = T.matrix("reduced rates","float32")#data

    # Convert rate_vector of shape [NZ, NB, 2N] to an array of shape INSHAPE
    red_R_fake = subsample_neurons(rate_vector, N=N, NZ=NZ, NB=NB,
                                   sample_sites=sample_sites,
                                   track_offset_identity=track_offset_identity)

    get_reduced = theano.function([rate_vector],red_R_fake,allow_input_downcast = True)

    ##Input Variable Definition

    in_fake = T.log(1. + red_R_fake)
    in_true = T.log(1. + red_R_true)

    #I want to make a network that takes red_R and gives a scalar output

    discriminator = SD.make_net(INSHAPE, "WGAN", LAYERS,
                                normalization=disc_normalization)

    #get the outputs
    true_dis_out = lasagne.layers.get_output(discriminator, in_true)
    fake_dis_out = lasagne.layers.get_output(discriminator, in_fake)

    D_acc = theano.function([rate_vector,red_R_true],fake_dis_out.mean() - true_dis_out.mean(),allow_input_downcast = True)

    #make the loss functions
    
    red_fake_for_grad = T.matrix("reduced rates","float32")#data
    in_for_grad = T.log(1. + red_fake_for_grad)

    for_grad_out = lasagne.layers.get_output(discriminator, in_for_grad)

    lam = WGAN_lambda

    DGRAD = T.jacobian(T.reshape(for_grad_out,[-1]),red_fake_for_grad).norm(2, axis = [1,2])#the norm of the gradient

    true_loss_exp = fake_dis_out.mean() - true_dis_out.mean() + lam*((DGRAD - 1)**2).mean()#discriminator loss
    fake_loss_exp = -fake_dis_out.mean()#generative loss

    if rate_penalty_no_I:
        penalized_rate = rate_vector[:, :, :N]
    else:
        penalized_rate = rate_vector
    fake_loss_exp_train = fake_loss_exp + rate_cost * SSgrad.rectify(penalized_rate - rate_penalty_threshold).mean()

    #make loss functions
    true_loss = theano.function([red_R_true,rate_vector,red_fake_for_grad],true_loss_exp,allow_input_downcast = True)
    fake_loss = theano.function([rate_vector],fake_loss_exp,allow_input_downcast = True)

    #Computing the G gradient!
    #
    #We have to do this by hand because hte SS soluytion is not written in a way that theano can solve
    #
    #to get the grads w.r.t. the generators parameters we need to do a jacobian 
    fake_dis_grad = T.jacobian(T.flatten(fake_loss_exp_train),rate_vector) #gradient of generator loss w.r.t rates
    fake_dis_grad = T.reshape(fake_dis_grad,[NZ,NB,2*N])

    #reshape the generator gradient to fit with Dr/Dth
    fake_dis_grad_expanded = T.reshape(fake_dis_grad,[NZ,NB,2*N,1,1])

    #putthem together and sum of the z,b,and N axes to get a [2,2] tensor that is the gradient of the loss w.r.t. parameters
    dLdJ_exp = (fake_dis_grad_expanded*R_grad[0]).sum(axis = (0,1,2))
    dLdD_exp = (fake_dis_grad_expanded*R_grad[1]).sum(axis = (0,1,2))
    dLdS_exp = (fake_dis_grad_expanded*R_grad[2]).sum(axis = (0,1,2))

    dLdJ = theano.function([rate_vector,ivec,Z],dLdJ_exp,allow_input_downcast = True)
    dLdD = theano.function([rate_vector,ivec,Z],dLdD_exp,allow_input_downcast = True)
    dLdS = theano.function([rate_vector,ivec,Z],dLdS_exp,allow_input_downcast = True)
    ####

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    b1 = .5
    b2 = .9

    G_updates = lasagne.updates.adam([dLdJ_exp,dLdD_exp,dLdS_exp],[J,D,S], g_lr,beta1 = b1,beta2 = b2)
    D_updates = lasagne.updates.adam(true_loss_exp,lasagne.layers.get_all_params(discriminator, trainable=True), d_lr,beta1 = b1,beta2 = b2)#discriminator training function

    G_train_func = theano.function([rate_vector,ivec,Z],fake_loss_exp,updates = G_updates,allow_input_downcast = True)
    D_train_func = theano.function([rate_vector,red_R_true,red_fake_for_grad],true_loss_exp,updates = D_updates,allow_input_downcast = True)

    G_loss_func = theano.function([rate_vector,ivec,Z],fake_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')
    D_loss_func = theano.function([rate_vector,red_R_true,red_fake_for_grad],true_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')

    return G_train_func, G_loss_func, D_train_func, D_loss_func,D_acc,get_reduced,discriminator


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument(
        '--iterations', default=100000, type=int,
        help='Number of iterations (default: %(default)s)')
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for random numbers (default: %(default)s)')
    parser.add_argument(
        '--init-disturbance', default=0.5, type=eval,
        help='''Initial disturbance to the parameter.  If it is
        evaluated to be a 3-tuple, the components are used for the
        disturbance for J, D (delta), S (sigma), respectively.  It
        accepts any Python expression.
        (default: %(default)s)''')
    parser.add_argument(
        '--gen-learn-rate', default=0.001, type=float,
        help='Learning rate for generator (default: %(default)s)')
    parser.add_argument(
        '--disc-learn-rate', default=0.001, type=float,
        help='Learning rate for discriminator (default: %(default)s)')
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
        '--sample-sites', default=[0], type=utils.csv_line(float),
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
        '--contrast',
        default=[5, 10, 30],
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
        '--truth_seed', default=42, type=int,
        help='Seed for generating ground truth data (default: %(default)s)')
    parser.add_argument(
        '--loss', default="WD",
        choices=('WD', 'CE', 'LS'),
        help='''Type of loss to use. Wasserstein Distance (WD), Cross-Entropy
        ("CE") or LSGAN ("LS"). (default: %(default)s)''')
    parser.add_argument(
        '--layers', default=[], type=eval,
        help='List of nnumbers of units in hidden layers (default: %(default)s)')
    parser.add_argument(
        '--n_samples', default=15, type=eval,
        help='''Number of samples to draw from G each step
        (aka NZ, minibatch size). (default: %(default)s)''')
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
        '--WGAN_lambda', default=10.0, type=float,
        help='The complexity penalty for the D (default: %(default)s)')
    parser.add_argument(
        '--WGAN_n_critic0', default=50, type=int,
        help='First critic iterations (default: %(default)s)')
    parser.add_argument(
        '--disc-normalization', default='none', choices=('none', 'layer'),
        help='Normalization used for discriminator.')
    parser.add_argument(
        '--disc-param-save-interval', default=5, type=int,
        help='''Save parameters for discriminator for each given
        generator step. -1 means to never save.
        (default: %(default)s)''')
    parser.add_argument(
        '--disc-param-template', default='last.npz',
        help='''Python string format for the name of the file to save
        discriminator parameters.  First argument "{}" to the format
        is the number of generator updates.  Not using "{}" means to
        overwrite to existing file (default) which is handy if you are
        only interested in the latest parameter.  Use "{}.npz" for
        recording the history of evolution of the discriminator.
        (default: %(default)s)''')
    parser.add_argument(
        '--disc-param-save-on-error', action='store_true',
        help='''Save discriminator parameter just before something
        when wrong (e.g., having NaN).
        (default: %(default)s)''')

    parser.add_argument(
        '--quiet', action='store_true',
        help='Do not print loss values per epoch etc.')

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
    ns = parser.parse_args(args)
    do_learning(learn, vars(ns))


def add_learning_options(parser):
    parser.add_argument(
        '--n_bandwidths', default=4, type=int, choices=(4, 5, 8),
        help='Number of bandwidths (default: %(default)s)')
    parser.add_argument(
        '--load-gen-param',
        help='''Path to generator.csv whose last row is loaded as the
        starting point.''')

    execution.add_base_learning_options(parser)


def preprocess(run_config):
    # Set `bandwidths` outside the `learn` function, so that
    # `bandwidths` is stored in info.json:
    n_bandwidths = run_config.pop('n_bandwidths')
    if n_bandwidths == 4:
        bandwidths = [0.0625, 0.125, 0.25, 0.75]
    elif n_bandwidths == 5:
        bandwidths = [0.0625, 0.125, 0.25, 0.5, 0.75]
    elif n_bandwidths == 8:
        bandwidths = [0, 0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1]
    else:
        raise ValueError('Unknown number of bandwidths: {}'
                         .format(n_bandwidths))
    run_config['bandwidths'] = bandwidths

    # Set initial parameter set J/D/S for generator.
    load_gen_param = run_config.pop('load_gen_param')
    if load_gen_param:
        lastrow = np.loadtxt(load_gen_param, delimiter=',')[-1]
        if len(lastrow) == 13:
            lastrow = lastrow[1:]
        elif len(lastrow) == 12:
            pass
        else:
            raise ValueError('Invalid format.'
                             ' Last row of {} contains {} columns.'
                             ' It has to contain 13 (or 12) columns.'
                             .format(load_gen_param, len(lastrow)))
        J0, D0, S0 = np.exp(lastrow).reshape((3, 2, 2)).tolist()
    else:
        J0 = [[0.0957, 0.0638], [0.1197, 0.0479]]
        D0 = [[0.7660, 0.5106], [0.9575, 0.3830]]
        S0 = [[0.08333375, 0.025], [0.166625, 0.025]]
    run_config.update(J0=J0, D0=D0, S0=S0)


def do_learning(learn, run_config):
    """
    Wrap `.execution.do_learning` with some pre-processing.
    """
    execution.do_learning(
        learn, run_config,
        preprocess=preprocess,
        extra_info=dict(
            n_bandwidths=run_config['n_bandwidths'],
            load_gen_param=run_config['load_gen_param'],
        ))


if __name__ == "__main__":
    main()
