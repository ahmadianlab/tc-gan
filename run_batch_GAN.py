"""
Run SSN-GAN learning given a MATLAB data file.
"""

import theano
import theano.tensor as T
import lasagne
import numpy as np
import scipy.io

import discriminators.simple_discriminator as SD
import nips_madness.gradient_expressions.make_w_batch as make_w
import nips_madness.gradient_expressions.SS_grad as SSgrad
import nips_madness.ssnode as SSsolve

import time

import stimuli

USEDATA = False
LAYERS = [128]
io_type = "asym_tanh"

def main(datapath, iterations, seed=1, gen_learn_rate=0.001, disc_learn_rate=0.001):

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
    n_sites = 10
    coe_value = float(mat['Modelparams'][0, 0]['k'][0, 0])
    exp_value = float(mat['Modelparams'][0, 0]['n'][0, 0])
    data = mat['E_Tuning']      # shape: (N_data, nb)


    #defining all the parameters that we might want to train

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
    dp = -.5

    J = theano.shared(J2.get_value() + dp*np.array([[1.,1.],[1.,1.]]),name = "j")
    D = theano.shared(D2.get_value() + dp*np.array([[1.,1.],[1.,1.]]),name = "d")
    S = theano.shared(S2.get_value() + dp*np.array([[1.,1.],[1.,1.]]),name = "s")

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
    nz = theano.shared(1,name = 'n_samples')
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
    dRdJ_exp = SSgrad.WRgrad_batch(rvec,ww,dwdj,ivec,exp,coe,NZ,NB,N,io_type)
    dRdD_exp = SSgrad.WRgrad_batch(rvec,ww,dwdd,ivec,exp,coe,NZ,NB,N,io_type)
    dRdS_exp = SSgrad.WRgrad_batch(rvec,ww,dwds,ivec,exp,coe,NZ,NB,N,io_type)

    dRdJ = theano.function([rvec,ivec,Z],dRdJ_exp,allow_input_downcast = True)
    dRdD = theano.function([rvec,ivec,Z],dRdD_exp,allow_input_downcast = True)
    dRdS = theano.function([rvec,ivec,Z],dRdS_exp,allow_input_downcast = True)

    #run gradient descent on W (minimize W*W)
    testDW = False
    if testDW:        
        testz = Ztest
        testI = np.random.normal(0,10,(NB,2*N)).astype("float32")
        
        wtest = W(testz)
        ssR = np.asarray([[SSsolve.solve_dynamics(2.,wtest[z],testI[b],k = coe_value, n = exp_value).astype("float32") for b in range(len(testI))] for z in range(len(testz))])
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
    DRtest = True
    if DRtest:
        print("starting DR")
    #I want to test this by adjusting the parameters to give some specified output
         
        inp = BAND_IN
        Ztest = np.random.rand(NZ,2*N,2*N)
        
        WW = W(Ztest)
               
        rr1 = np.array([[SSsolve.solve_dynamics(1.,z,b,r0 = np.zeros((2*N)),k = coe_value,n = exp_value) for b in inp] for z in WW])

        print("Computing symbolic gradient")
        DR =[dRdJ(rr1,inp,Ztest),dRdD(rr1,inp,Ztest),dRdS(rr1,inp,Ztest)]
               
        dd = 0.0001

        J.set_value(J.get_value() + np.array([[dd,0],[0,0]]))
        #D.set_value(D.get_value() + np.array([[dd,0],[0,0]]))
        #S.set_value(S.get_value() + np.array([[dd,0.],[0.,0.]]))

        WW = W(Ztest)

        rr2 = np.array([[SSsolve.solve_dynamics(1.,WW[z],inp[b],r0 = rr1[z,b],k = coe_value,n = exp_value) for b in range(len(inp))] for z in range(len(WW))])

        print(rr2.max())
        print(rr2.min())

        print(rr1.max())
        print(rr1.min())

        DRa = (rr2 - rr1)/dd
        DRp = DR[0][:,:,:,0,0]

        print(np.max(np.abs(DRa-DRp))/np.max(np.abs(DRa)))
 
        exit()

    ###Now I need to make the GAN
    ###
    ###
    ###    
    ###I want to make a network that takes a tensor of shape [2N] and generates dl/dr

    red_R_true = T.matrix("reduced rates","float32")#data

    red_R_fake = T.reshape(T.tensordot(rvec,M,axes = [2,1]),[NZ,NB])#generated by our generator function
    get_reduced = theano.function([rvec],red_R_fake,allow_input_downcast = True)

    #Defines the input shape for the discriminator network
    INSHAPE = [NZ,NB]
 
    #I want to make a network that takes red_R and gives a scalar output

    DIS_red_r_true = SD.make_net(red_R_true,INSHAPE,LAYERS)
    DIS_red_r_fake = SD.make_net(red_R_fake,INSHAPE,LAYERS,params = lasagne.layers.get_all_layers(DIS_red_r_true))

    #get the outputs
    true_dis_out = lasagne.layers.get_output(DIS_red_r_true)
    fake_dis_out = lasagne.layers.get_output(DIS_red_r_fake)

    D_acc = theano.function([rvec,red_R_true],(true_dis_out.sum() + (1 - fake_dis_out).sum())/(2*NZ),allow_input_downcast = True)

    #make the loss functions
    if 0:
        SM = .8
        true_loss_exp = -SM*np.log(true_dis_out).mean() - (1. - SM)*np.log(1. - true_dis_out).mean() - np.log(1. - fake_dis_out).mean()#discriminator loss
        fake_loss_exp = -np.log(fake_dis_out).mean()#generative loss
    else:
        true_loss_exp = ((true_dis_out - 1.)**2).mean() + ((fake_dis_out + 1.)**2).mean()#discriminator loss
        fake_loss_exp = ((fake_dis_out - 1.)**2).mean()#generative loss

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    D_updates = lasagne.updates.adam(true_loss_exp,lasagne.layers.get_all_params(DIS_red_r_true), disc_learn_rate)#discriminator training function

    #make loss functions
    true_loss = theano.function([red_R_true,rvec],true_loss_exp,allow_input_downcast = True)
    fake_loss = theano.function([rvec],fake_loss_exp,allow_input_downcast = True)

    #to get the grads w.r.t. the generators parameters we need to do a jacobian 
    fake_dis_grad = T.jacobian(T.flatten(fake_loss_exp),rvec) #gradient of generator loss w.r.t rates
    fake_dis_grad = T.reshape(fake_dis_grad,[NZ,NB,2*N])

    t_grad = theano.function([rvec],fake_dis_grad,allow_input_downcast = True)#gradient function

    #reshape the generator gradient to fit with Dr/Dth
    fake_dis_grad_expanded = T.reshape(fake_dis_grad,[NZ,NB,2*N,1,1])

    #putthem together and sum of the z,b,and N axes to get a [2,2] tensor that is the gradient of the loss w.r.t. parameters
    dLdJ_exp = (fake_dis_grad_expanded*dRdJ_exp).sum(axis = (0,1,2))
    dLdD_exp = (fake_dis_grad_expanded*dRdD_exp).sum(axis = (0,1,2))
    dLdS_exp = (fake_dis_grad_expanded*dRdS_exp).sum(axis = (0,1,2))

    dLdJ = theano.function([rvec,ivec,Z],dLdJ_exp,allow_input_downcast = True)
    dLdD = theano.function([rvec,ivec,Z],dLdD_exp,allow_input_downcast = True)
    dLdS = theano.function([rvec,ivec,Z],dLdS_exp,allow_input_downcast = True)

    G_updates = lasagne.updates.adam([dLdJ_exp,dLdD_exp,dLdS_exp],[J,D,S], gen_learn_rate)

    G_train_func = theano.function([rvec,ivec,Z],fake_loss_exp,updates = G_updates,allow_input_downcast = True)
    D_train_func = theano.function([rvec,red_R_true],true_loss_exp,updates = D_updates,allow_input_downcast = True)

    G_loss_func = theano.function([rvec,ivec,Z],fake_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')
    D_loss_func = theano.function([rvec,red_R_true],true_loss_exp,allow_input_downcast = True,on_unused_input = 'ignore')

    #Now we set up values to use in testing.

    inp = BAND_IN

    def log(a,F = "./SSNGAN_log_{}.log".format(len(LAYERS)),PRINT = True):
        if PRINT:
            print(a)
        f = open(F,"a")
        f.write(str(a) + "\n")
        f.close()

    log("epoch,Gloss,Dloss,Daccuracy,SSsolve_time,gradient_time,model_convergence,truth_convergence")

    for k in range(iterations):
        TT = time.time()
        rz = np.zeros([2*N])

        ####
        np.random.shuffle(data)

        #the data
        true = data[:NZ]
        ####

        #generated samples
        #
        #This chunk of code generates samples from teh fitted model adn runs the G update
        #
        #
        ###################
        Ftest = []
        Ztest = []
        model_fail = 0
        while len(Ftest) < NZ:
            ztest = np.random.rand(1,2*N,2*N) 
            wtest = W(ztest)
            rates = [SSsolve.solve_dynamics(.1,wtest[0],inp[i],r0 = rz,
                                             k = coe_value, n = exp_value)
                     for i in range(len(inp))]

            if np.all(np.isfinite(rates)):
                Ftest.append(rates)
                Ztest.append(ztest[0])
            else:
                model_fail += 1

        Ftest = np.array(Ftest)
        Ztest = np.array(Ztest)
        rtest = np.array([[c for c in TC] for TC in Ftest])
        stest = np.array([[c for c in TC] for TC in Ftest])
        
        Tfinal = time.time()

        Gloss = G_train_func(rtest,inp,Ztest)

        ########################################
        ########################################
        ########################################
        ########################################

        #True model generator and D train
        #
        #This part generates the "true" TC and updates D
        #
        #
        #
        #################################

        
        ##faketest
        if USEDATA == False:
            Otrue = []
            Ztrue = []
            true_fail = 0

            while len(Otrue) < NZ:
                ztest2 = np.random.rand(1,2*N,2*N) 
                
                wtest2 = W_test(ztest2)
                
                rates = [SSsolve.solve_dynamics(.1,wtest2[0],inp[i],r0 = rz,
                                                 k = coe_value, n = exp_value)
                         for i in range(len(inp))]
                
                if np.all(np.isfinite(rates)):
                    Otrue.append(rates)
                    Ztrue.append(ztest2[0])
                else:
                    true_fail += 1

            Otrue = np.array(Otrue)
            Ztrue = np.array(Ztrue)
            true = get_reduced(np.array([[O for O in TC] for TC in Otrue]))                
 
        else:
            true_fail = 0
            
        Dloss = D_train_func(rtest,true)

        ###################################
        ###################################
        
        log("{},{},{},{},{},{},{},{}".format(k,Gloss,Dloss,D_acc(rtest,true),Tfinal - TT,time.time() - Tfinal,model_fail,true_fail))

        GZmean = get_reduced(rtest).mean(axis = 0)
        Dmean = true.mean(axis = 0)
            
        Dparam = lasagne.layers.get_all_layers(DIS_red_r_true)[-1]
        
        if len(LAYERS) == 0:
            DW = Dparam.W.get_value()
            DB = Dparam.b.get_value()
        else:
            DW = np.ones((8,1))
            DB = [0.]

            
        logstrG = "{},{},{},{},{},{},{},{}".format(GZmean[0],GZmean[1],GZmean[2],GZmean[3],GZmean[4],GZmean[5],GZmean[6],GZmean[7])
        logstrD = "{},{},{},{},{},{},{},{}".format(Dmean[0],Dmean[1],Dmean[2],Dmean[3],Dmean[4],Dmean[5],Dmean[6],Dmean[7])
        logstrW = "{},{},{},{},{},{},{},{},{}".format(DW[0,0],DW[1,0],DW[2,0],DW[3,0],DW[4,0],DW[5,0],DW[6,0],DW[7,0],DB[0])
        
        log(logstrG + "," + logstrD + "," + logstrW,F = "D_parameters_{}.log".format(len(LAYERS)),PRINT = False) 
        
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

            log(string,F = "./parameters_{}.log".format(len(LAYERS)),PRINT = False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'datapath', default='training_data_TCs_Ne102.mat', nargs='?',
        help='Path to MATLAB data file (default: %(default)s)')
    parser.add_argument(
        '--iterations', default=10000, type=int,
        help='Number of iterations (default: %(default)s)')
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for random numbers (default: %(default)s)')
    parser.add_argument(
        '--gen-learn-rate', default=0.01, type=float,
        help='Learning rate for generator (default: %(default)s)')
    parser.add_argument(
        '--disc-learn-rate', default=0.01, type=float,
        help='Learning rate for discriminator (default: %(default)s)')
    ns = parser.parse_args()
    main(**vars(ns))
