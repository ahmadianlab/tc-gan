"""

Run SSN-GAN learning given a MATLAB data file.

"""

import theano
import theano.tensor as T
import lasagne
import numpy as np
import scipy.io

from nips_madness import utils
import discriminators.simple_discriminator as SD
import nips_madness.gradient_expressions.make_w_batch as make_w
import nips_madness.gradient_expressions.SS_grad as SSgrad
import nips_madness.ssnode as SSsolve

import json
import os
import time

import stimuli


def main(datapath, iterations, seed, gen_learn_rate, disc_learn_rate,
         loss, use_data, IO_type, layers, n_samples, debug, rate_cost, WGAN,
         rate_hard_bound, rate_soft_bound,
         run_config):
    meta_info = utils.get_meta_info(packages=[np, scipy, theano, lasagne])

    if WGAN:
        make_functions = make_WGAN_functions
        train_update = WGAN_update
    else:
        make_functions = make_RGAN_functions
        train_update = RGAN_update


    ##Make the tag for the files
    #tag whether or not we use data
    if use_data:
        tag = "data_"
    else:
        tag = "generated_"

    #tag the IO and loss
    tag = tag + IO_type + "_" + loss

    #tag the layers
    for l in layers:
        tag = tag + "_" + str(l)

    #tag the rate cost
    if float(rate_cost) > 0:
        tag = tag + "_" + str(rate_cost)
    rate_cost = float(rate_cost)

    if WGAN:
        tag = tag + "_WGAN"
    else:
        tag = tag + "_RGAN"
        
    print(tag)
    #if debug mode, throw that all away
    if debug:
        tag = "DEBUG"
    ############################


    np.random.seed(seed)

    # Load data and "experiment" parameter (note that all [0]s are to
    # get rid of redundant dimensions of the MATLAB data and not
    # throwing away data):
    mat = scipy.io.loadmat(datapath)
    L_mat = mat['Modelparams'][0, 0]['L'][0, 0]
    bandwidths = mat['Modelparams'][0, 0]['bandwidths'][0] / L_mat
    smoothness = mat['Modelparams'][0, 0]['l_margin'][0, 0] / L_mat
    contrast = mat['Modelparams'][0, 0]['c'][0, 0]
    n_sites = int(mat['Modelparams'][0, 0]['Ne'][0, 0])
    coe_value = float(mat['Modelparams'][0, 0]['k'][0, 0])
    exp_value = float(mat['Modelparams'][0, 0]['n'][0, 0])
    data = mat['E_Tuning']      # shape: (N_data, nb)

    ssn_params = dict(
        rate_soft_bound=rate_soft_bound,
        rate_hard_bound=rate_hard_bound,
        io_type=IO_type,
        k=coe_value,
        n=exp_value,
        r0=np.zeros(2 * n_sites),
    )

    #defining all the parameters that we might want to train

    print(n_sites)

    exp = theano.shared(exp_value,name = "exp")
    coe = theano.shared(coe_value,name = "coe")

    #these are parameters we will use to test the GAN
    J2 = theano.shared(np.log(np.array([[.0957,.0638],[.1197,.0479]])).astype("float64"),name = "j")
    D2 = theano.shared(np.log(np.array([[.7660,.5106],[.9575,.3830]])).astype("float64"),name = "d")
    S2 = theano.shared(np.log(np.array([[.6667,.2],[1.333,.2]])/L_mat).astype("float64"),name = "s")

#    J = theano.shared(np.log(np.array([[1.5,.25],[2.0,.2]])).astype("float64"),name = "j")
#    D = theano.shared(np.log(np.array([[5.,2.],[18.,1.5]])).astype("float64"),name = "d")
#    S = theano.shared(np.log(np.array([[.5,.3],[1.0,.25]])).astype("float64"),name = "s")

    Jp2 = T.exp(J2)
    Dp2 = T.exp(D2)
    Sp2 = T.exp(S2)

    #these are the parammeters to be fit
    dp = .5

    J = theano.shared(J2.get_value() + dp*np.array([[1,1],[1,1]]),name = "j")
    D = theano.shared(D2.get_value() + dp*np.array([[1,1],[1,1]]),name = "d")
    S = theano.shared(S2.get_value() + dp*np.array([[1,1],[1,1]]),name = "s")

#    J = theano.shared(np.log(np.array([[1.5,.25],[2.0,.2]])).astype("float64"),name = "j")
#    D = theano.shared(np.log(np.array([[5.,2.],[18.,1.5]])).astype("float64"),name = "d")
#    S = theano.shared(np.log(np.array([[.5,.3],[1.0,.25]])).astype("float64"),name = "s")

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
    nb = theano.shared(data.shape[1],name = 'n_stim')

    #array that computes the positions
    X = theano.shared(np.linspace(-.5,.5,n.get_value()).astype("float32"),name = "positions")

    ##getting regular nums##
    N = int(n.get_value())
    NZ = int(nz.get_value())
    NB = int(nb.get_value())
    m = 1
    ###


    BAND_IN = stimuli.input(bandwidths, X.get_value(), smoothness, contrast)

    #a mask to get from the rates to just the ones we are measuring
    M = theano.shared(np.array([[1 if k == j + (N/2) - (m/2) else 0 for k in range(2*N)] for j in range(m)]).astype("float32"),"mask")

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

    convtest = False
    if convtest:
        zt = np.random.rand(1,2*N,2*N)
        wt = W_test(zt)[0]
        rz = np.zeros((2*N))

        for c in [1.,2.,4.,8.,16.]:
            r  = SSsolve.fixed_point(wt,c*BAND_IN[-1], *ssn_params)
        
            print(np.max(r.x))
        exit()

    timetest = False
    if timetest:
        times = []
        EE = 0
        for k in range(100):
            zt = np.random.rand(1,2*N,2*N)
            wt = W_test(zt)[0]
            rz = np.zeros((2*N))

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
    testDW = False
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
    DRtest = False
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
    
    G_train_func,G_loss_func,D_train_func,D_loss_func,D_acc,get_reduced,DIS_red_r_true = make_functions(
        rate_vector=rvec, mask=M, NZ=NZ, NB=NB, LOSS=loss, LAYERS=layers,
        d_lr=disc_learn_rate, g_lr=gen_learn_rate, rate_cost=rate_cost,
        ivec=ivec, Z=Z, J=J, D=D, S=S, N=N,
        R_grad=[dRdJ_exp, dRdD_exp, dRdS_exp])

    #Now we set up values to use in testing.

    inp = BAND_IN


    def log(a,F = "SSNGAN_log_{}.log".format(tag),PRINT = True):
        if isinstance(a, list):
            a = ','.join(map(str, a))
        if PRINT:
            print(a)
        f = open("./logfiles/" + F,"a")
        f.write(str(a) + "\n")
        f.close()

    try:
        os.makedirs('logfiles')
    except OSError as err:
        if err.errno != 17:
            # If the error is "File exists" (errno=17) error, it means
            # that the directory exists and it's fine to ignore the
            # error.  It is slightly safer than checking existence of
            # the directory since several processes may be creating it
            # at the same time.  Maybe the error should be re-raised
            # here but opening log file would fail anyway if there is
            # something wrong.
            print("!! Unexpected exception !!")
            print(err)

    with open(os.path.join('logfiles', 'info_{}.json'.format(tag)), 'w') as fp:
        json.dump(dict(
            run_config=run_config,
            meta_info=meta_info,
        ), fp)

    # Clear data in old log files
    for filename in ["SSNGAN_log_{}.log",
                     "D_parameters_{}.log",
                     "parameters_{}.log"]:
        open(os.path.join("logfiles", filename.format(tag)), 'w').close()

    log("epoch,Gloss,Dloss,Daccuracy,SSsolve_time,gradient_time,model_convergence,truth_convergence,model_unused,truth_unused")

    
    for k in range(iterations):
        TT = time.time()
        rz = np.zeros([2*N])

        Dloss,Gloss,rtest,true,model_info,true_info = train_update(D_train_func,G_train_func,iterations,N,NZ,NB,data,W,W_test,inp,ssn_params,use_data,log,D_acc,get_reduced,DIS_red_r_true,tag,J,D,S,WG_repeat = 5)

        Tfinal = time.time()

        log([k, Gloss, Dloss, D_acc(rtest, true),
             Tfinal - TT,
             time.time() - Tfinal,
             model_info.rejections,
             true_info.rejections,
             model_info.unused,
             true_info.unused])

        GZmean = get_reduced(rtest).mean(axis = 0)
        Dmean = true.mean(axis = 0)
 
        Dparam = lasagne.layers.get_all_layers(DIS_red_r_true)[-1]
        
        if len(layers) == 0:
            DW = Dparam.W.get_value()
            DB = Dparam.b.get_value()
        else:
            DW = np.ones((8,1))
            DB = [0.]

            
        logstrG = "{},{},{},{},{},{},{},{}".format(GZmean[0],GZmean[1],GZmean[2],GZmean[3],GZmean[4],GZmean[5],GZmean[6],GZmean[7])
        logstrD = "{},{},{},{},{},{},{},{}".format(Dmean[0],Dmean[1],Dmean[2],Dmean[3],Dmean[4],Dmean[5],Dmean[6],Dmean[7])
        logstrW = "{},{},{},{},{},{},{},{},{}".format(DW[0,0],DW[1,0],DW[2,0],DW[3,0],DW[4,0],DW[5,0],DW[6,0],DW[7,0],DB[0])
        
        log(logstrG + "," + logstrD + "," + logstrW,F = "D_parameters_{}.log".format(tag),PRINT = False) 
        
        if k%1 == 0:
            jj = J.get_value()
            dd = D.get_value()
            ss = S.get_value()
            
            allpar = np.reshape(np.concatenate([jj,dd,ss]),[-1]).tolist()

            string = "{},{},{},{},{},{},{},{},{},{},{},{},{}".format(k,
                                                                     allpar[0],
                                                                     allpar[1],
                                                                     allpar[2],
                                                                     allpar[3],
                                                                     allpar[4],
                                                                     allpar[5],
                                                                     allpar[6],
                                                                     allpar[7],
                                                                     allpar[8],
                                                                     allpar[9],
                                                                     allpar[10],
                                                                     allpar[11])

            log(string,F = "./parameters_{}.log".format(tag),PRINT = False)

 
def WGAN_update(D_train_func,G_train_func,iterations,N,NZ,NB,data,W,W_test,inp,ssn_params,use_data,log,D_acc,get_reduced,DIS_red_r_true,tag,J,D,S,WG_repeat = 5):

    for rep in range(WG_repeat):
        ####
        np.random.shuffle(data)
        
        #the data
        true = data[:NZ]
        ####
            
        #generated samples
        #
        #This chunk of code generates samples from teh fitted model adn runs the G update
        #
        ###################
        def Z_W_gen():
            while True:
                ztest = np.random.rand(1, 2*N, 2*N)
                wtest, = W(ztest)
                yield ztest[0], wtest
                    
        Ztest, Ftest, model_info = SSsolve.find_fixed_points(
            NZ, Z_W_gen(), inp,
            **ssn_params)
        
        Ftest = np.array(Ftest)
        Ztest = np.array(Ztest)
        rtest = np.array([[c for c in TC] for TC in Ftest])

        #True model generator and D train
        #
        #This part generates the "true" TC and updates D
        #
        #################################

        if use_data == False:
            def Z_W_gen2():
                while True:
                    ztest2 = np.random.rand(1, 2*N, 2*N)
                    wtest2, = W_test(ztest2)
                    yield ztest2[0], wtest2
                    
            Ztrue, Otrue, true_info = SSsolve.find_fixed_points(
                NZ, Z_W_gen2(), inp,
                **ssn_params)
            
            Otrue = np.array(Otrue)
            Ztrue = np.array(Ztrue)
            true = get_reduced(np.array([[O for O in TC] for TC in Otrue]))                
            
        else:
            true_info = SSsolve.null_FixedPointsInfo
            

        ###################################
        ###################################

        eps = np.random.rand(NZ, 1)
        
        Dloss = D_train_func(rtest,true,eps*true + (1. - eps)*get_reduced(rtest))

    #end D loop

    Gloss = G_train_func(rtest,inp,Ztest)
    
    return Dloss,Gloss,rtest,true,model_info,true_info

def RGAN_update(D_train_func,G_train_func,iterations,N,NZ,NB,data,W,W_test,inp,ssn_params,use_data,log,D_acc,get_reduced,DIS_red_r_true,tag,J,D,S,WG_repeat = 5):

        ####
    np.random.shuffle(data)
    
        #the data
    true = data[:NZ]
        ####
            
        #generated samples
        #
        #This chunk of code generates samples from teh fitted model adn runs the G update
        #
        ###################
    def Z_W_gen():
        while True:
            ztest = np.random.rand(1, 2*N, 2*N)
            wtest, = W(ztest)
            yield ztest[0], wtest
                    
    Ztest, Ftest, model_info = SSsolve.find_fixed_points(
        NZ, Z_W_gen(), inp,
        **ssn_params)
        
    Ftest = np.array(Ftest)
    Ztest = np.array(Ztest)
    rtest = np.array([[c for c in TC] for TC in Ftest])
    
        #True model generator and D train
        #
        #This part generates the "true" TC and updates D
        #
        #################################

    if use_data == False:
        def Z_W_gen2():
            while True:
                ztest2 = np.random.rand(1, 2*N, 2*N)
                wtest2, = W_test(ztest2)
                yield ztest2[0], wtest2
                
        Ztrue, Otrue, true_info = SSsolve.find_fixed_points(
            NZ, Z_W_gen2(), inp,
            **ssn_params)
        
        Otrue = np.array(Otrue)
        Ztrue = np.array(Ztrue)
        true = get_reduced(np.array([[O for O in TC] for TC in Otrue]))                
        
    else:
        true_info = SSsolve.null_FixedPointsInfo
        
        ###################################
        ###################################

    Dloss = D_train_func(rtest,true)
    Gloss = G_train_func(rtest,inp,Ztest)
    
    return Dloss,Gloss,rtest,true,model_info,true_info
    
def make_RGAN_functions(rate_vector,mask,NZ,NB,LOSS,LAYERS,d_lr,g_lr,rate_cost,ivec,Z,J,D,S,N,R_grad):
    ###Now I need to make the GAN
    ###
    ###
    ###    
    ###I want to make a network that takes a tensor of shape [2N] and generates dl/dr

    red_R_true = T.matrix("reduced rates","float32")#data

    red_R_fake = T.reshape(T.tensordot(rate_vector,mask,axes = [2,1]),[NZ,NB])#generated by our generator function
    get_reduced = theano.function([rate_vector],red_R_fake,allow_input_downcast = True)

    #Defines the input shape for the discriminator network
    INSHAPE = [NZ,NB]
 
    ##Input Variable Definition

    in_fake = T.log(1. + red_R_fake)
    in_true = T.log(1. + red_R_true)

    #I want to make a network that takes red_R and gives a scalar output

    DIS_red_r_true = SD.make_net(in_true,INSHAPE,LOSS,LAYERS)
    DIS_red_r_fake = SD.make_net(in_fake,INSHAPE,LOSS,LAYERS,params = lasagne.layers.get_all_layers(DIS_red_r_true))

    #get the outputs
    true_dis_out = lasagne.layers.get_output(DIS_red_r_true)
    fake_dis_out = lasagne.layers.get_output(DIS_red_r_fake)

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

    fake_loss_exp_train = fake_loss_exp + rate_cost * SSgrad.rectify((rate_vector - 200.)/10).sum()**2

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    D_updates = lasagne.updates.adam(true_loss_exp,lasagne.layers.get_all_params(DIS_red_r_true), d_lr)#discriminator training function

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

    return G_train_func,G_loss_func,D_train_func,D_loss_func,D_acc,get_reduced,DIS_red_r_true

def make_WGAN_functions(rate_vector,mask,NZ,NB,LOSS,LAYERS,d_lr,g_lr,rate_cost,ivec,Z,J,D,S,N,R_grad):

    ###I want to make a network that takes a tensor of shape [2N] and generates dl/dr

    red_R_true = T.matrix("reduced rates","float32")#data

    red_R_fake = T.reshape(T.tensordot(rate_vector,mask,axes = [2,1]),[NZ,NB])#generated by our generator function

    get_reduced = theano.function([rate_vector],red_R_fake,allow_input_downcast = True)

    #Defines the input shape for the discriminator network
    INSHAPE = [NZ,NB]
 
    ##Input Variable Definition

    in_fake = T.log(1. + red_R_fake)
    in_true = T.log(1. + red_R_true)

    #I want to make a network that takes red_R and gives a scalar output

    DIS_red_r_true = SD.make_net(in_true,INSHAPE,"WGAN",LAYERS)
    DIS_red_r_fake = SD.make_net(in_fake,INSHAPE,"WGAN",LAYERS,params = lasagne.layers.get_all_layers(DIS_red_r_true))

    #get the outputs
    true_dis_out = lasagne.layers.get_output(DIS_red_r_true)
    fake_dis_out = lasagne.layers.get_output(DIS_red_r_fake)

    D_acc = theano.function([rate_vector,red_R_true],fake_dis_out.mean() - true_dis_out.mean(),allow_input_downcast = True)

    #make the loss functions
    
    red_fake_for_grad = T.matrix("reduced rates","float32")#data
    in_for_grad = T.log(1. + red_fake_for_grad)
    
    D_red_grad_net = SD.make_net(in_for_grad,INSHAPE,"WGAN",LAYERS,params = lasagne.layers.get_all_layers(DIS_red_r_true))
    for_grad_out = lasagne.layers.get_output(D_red_grad_net)

    lam = 10.

    DGRAD = T.jacobian(T.reshape(for_grad_out,[-1]),red_fake_for_grad).norm(2, axis = [1,2])#the norm of the gradient

    true_loss_exp = fake_dis_out.mean() - true_dis_out.mean() + lam*((DGRAD - 1)**2).mean()#discriminator loss
    fake_loss_exp = -fake_dis_out.mean()#generative loss
    
    fake_loss_exp_train = fake_loss_exp + rate_cost * SSgrad.rectify((rate_vector - 200.)/10).sum()**2

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
    D_updates = lasagne.updates.adam(true_loss_exp,lasagne.layers.get_all_params(DIS_red_r_true), d_lr,beta1 = b1,beta2 = b2)#discriminator training function

    G_train_func = theano.function([rate_vector,ivec,Z],fake_loss_exp,updates = G_updates,allow_input_downcast = True)
    D_train_func = theano.function([rate_vector,red_R_true,red_fake_for_grad],true_loss_exp,updates = D_updates,allow_input_downcast = True)

    G_loss_func = theano.function([rate_vector,ivec,Z],fake_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')
    D_loss_func = theano.function([rate_vector,red_R_true,red_fake_for_grad],true_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')

    return G_train_func, G_loss_func, D_train_func, D_loss_func,D_acc,get_reduced,DIS_red_r_true

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'datapath', nargs='?',
        default=os.path.join(os.path.dirname(__file__),
                             'training_data_TCs_Ne102.mat'),
        help='Path to MATLAB data file (default: %(default)s)')
    parser.add_argument(
        '--iterations', default=10000, type=int,
        help='Number of iterations (default: %(default)s)')
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for random numbers (default: %(default)s)')
    parser.add_argument(
        '--gen-learn-rate', default=0.0001, type=float,
        help='Learning rate for generator (default: %(default)s)')
    parser.add_argument(
        '--disc-learn-rate', default=0.0001, type=float,
        help='Learning rate for discriminator (default: %(default)s)')
    parser.add_argument(
        '--use-data', default=False, action='store_true',
        help='Use data (True) or generate our own TC samples (False) (default: %(default)s)')
    parser.add_argument(
        '--debug', default=False, action='store_true',
        help='Run in debug mode. Save logs with DEBUG tag')
    parser.add_argument(
        '--IO_type', default="asym_tanh",
        help='Type of nonlinearity to use. Regular ("asym_power"). Linear ("asym_linear"). Tanh ("asym_tanh") (default: %(default)s)')
    parser.add_argument(
        '--loss', default="CE",
        help='Type of loss to use. Cross-Entropy ("CE") or LSGAN ("LS"). (default: %(default)s)')
    parser.add_argument(
        '--layers', default=[], type=eval,
        help='List of nnumbers of units in hidden layers (default: %(default)s)')
    parser.add_argument(
        '--n_samples', default=10, type=eval,
        help='Number of samples to draw from G each step (default: %(default)s)')
    parser.add_argument(
        '--rate_cost', default='0',
        help='The cost of having the rate be large (default: %(default)s)')
    parser.add_argument(
        '--rate_soft_bound', default=200, type=float,
        help='rate_soft_bound=r0 (default: %(default)s)')
    parser.add_argument(
        '--rate_hard_bound', default=1000, type=float,
        help='rate_hard_bound=r1 (default: %(default)s)')
    parser.add_argument(
        '--WGAN', default=False, action='store_true',
        help='Use WGAN (default: %(default)s)')

    ns = parser.parse_args()

    # Collect all arguments/options in a dictionary, in order to save
    # it elsewhere:
    run_config = vars(ns)
    # ...then use it as keyword arguments
    main(run_config=run_config, **run_config)
