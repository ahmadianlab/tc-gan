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

def main(datapath, iterations, seed=0):
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

    #defining all the parameters that we might want to train

    exp = theano.shared(exp_value,name = "exp")
    coe = theano.shared(coe_value,name = "coe")

    J = theano.shared(np.array([[-.1,.5],[.5,-.2]]).astype("float64"),name = "j")
    D = theano.shared(np.array([[-1,-1],[-1,-1]]).astype("float64"),name = "d")
    S = theano.shared(np.array([[0,-1.],[.1,-1.]]).astype("float64"),name = "s")

    Jp = T.exp(J)
    Dp = T.exp(D)
    Sp = T.exp(S)

    #compute jacobian of the primed variables w.r.t. J,D,S.
    dJpJ = T.reshape(T.jacobian(T.reshape(Jp,[-1]),J),[2,2,2,2])
    dDpD = T.reshape(T.jacobian(T.reshape(Dp,[-1]),D),[2,2,2,2])
    dSpS = T.reshape(T.jacobian(T.reshape(Sp,[-1]),S),[2,2,2,2])

    #specifying the shape of model/input
    n = theano.shared(n_sites,name = "n_sites")
    nz = theano.shared(10,name = 'n_samples')
    nb = theano.shared(data.shape[1],name = 'n_stim')

    #array that computes the positions
    X = theano.shared(np.linspace(-.5,.5,n.get_value()).astype("float32"),name = "positions")


    ##getting regular nums##
    N = int(n.get_value())
    NZ = int(nz.get_value())
    NB = int(nb.get_value())
    m = 1
    ###

    BAND_IN = stimuli.input(bandwidths, X.get_value(), smoothness)

    #a mask to get from the rates to just the ones we are measuring
    M = theano.shared(np.array([[1 if k == j + (N/2) - (m/2) else 0 for k in range(2*N)] for j in range(m)]).astype("float32"),"mask")

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

    #get deriv. of W given Z
    DWj = theano.function([Z],dwdj,allow_input_downcast = True,on_unused_input = "ignore")
    DWd = theano.function([Z],dwdd,allow_input_downcast = True,on_unused_input = "ignore")
    DWs = theano.function([Z],dwds,allow_input_downcast = True,on_unused_input = "ignore")

    #a random Z sample for use in testing
    Ztest = np.random.rand(NZ,2*N,2*N).astype("float32")
    #now we need to get a function to generate dr/dth from dw/dth

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


    #run gradient descent on W (minimize W*W)
    testDW = False
    if testDW:        
        testz = Ztest
        testI = np.random.normal(0,10,(NB,2*N)).astype("float32")
        
        wtest = W(testz)
        ssR = np.asarray([[SSsolve.fixed_point(wtest[z],testI[b]).x.astype("float32") for b in range(len(testI))] for z in range(len(testz))])
        print(ssR.mean())
        print(wtest.mean())
        print(DWj(testz).mean())
        
        
        print("Start DWtest")
        for k in range(100):
            wtest = np.reshape(W(testz),[NZ,2*N,2*N,1,1])
            

            dd = DWj(testz)#[nz,2N,2N,2,2]
            dj = 2*(wtest*dd).mean(axis = 0).mean(axis = 0).mean(axis = 0)
            
            jnew = J.get_value() - .001*dj
            
            J.set_value(jnew.astype("float32"))
            
            if k % 1 == 0:
                print((wtest*wtest).mean())

        exit()
        #running this will verify that the W gradient is working (by adjusting parameters to minimize the mean sqaured W entry)


    print("defined all the functions")

    def dR_dtheta(inp,ZZ):
        '''

        I need a function which takes 

          inp - [nb,2N]
          Z - [nz,2N,2N]

        and generates dr/dth - [3,nz,nb,2N,2,2]

        '''

        if len(ZZ.shape) != 3:
            print("Z must have three dimentions - [nz,2N,2N]")

        if len(inp.shape) != 2:
            print("inp must have two dimentions - [nb,2N]")

        T1 = time.time()
        #[nz,2n,2n]
        WW = W(ZZ)

        T2 = time.time()
        #[nz,nb,2N]
        RR = np.array([[SSsolve.fixed_point(WW[z],inp[i]).x for i in range(len(inp))] for z in range(len(ZZ))])

        T3 = time.time()
        DRTj = dRdJ(RR,inp,ZZ)
        DRTd = dRdD(RR,inp,ZZ)
        DRTs = dRdS(RR,inp,ZZ)
 
        T4 = time.time()
        
        DR = np.array([DRTj,DRTd,DRTs])

#        print("{}\t{}\t{}".format(T2-T1,T3-T2,T4-T3))

        return DR


    timetest = False
    if timetest:
        zz = np.random.rand(NZ,2*N,2*N).astype("float32")
        II = np.random.normal(0,1,[NB,2*N]).astype("float32")
        print("Computing gradients")

        T_temp = time.time()
        dwt = DWj(zz)
        print("dwt took {} seconds".format(time.time() - T_temp))
        
        T_temp = time.time()
 
        DR = dR_dtheta(II,zz)
        print(DR.mean())
        
        print("A single datapoint took {} seconds to get dRdTH".format(time.time() - T_temp))
        print("An 8*5 batch would take {} seconds to get dRdTH".format(8*5*(time.time() - T_temp)))
        
        print(DR.shape)
        exit()

    #do gradient descent on R
    DRtest = False
    if DRtest:
    #I want to test this by adjusting the parameters to give some specified output
        
        def lossF(i,z,tar):
            WW = W(z)
            
            r = np.array([[SSsolve.fixed_point(WW[n],i[m]).x for m in range(len(i))] for n in range(len(z))])
            
            return ((r - tar)**2).mean(),r
        
    ##If the target is tar, and the loss is sqred loss, then Dl/Dr = 2*(r - tar)
    ##so Dl/Dt = 2*(r - tar).Dr/Dth

        
        target = np.random.normal(2,.1,(NZ,NB,2*N))
#        target = np.ones_like(target)
        
        inp = np.random.normal(0,1,[NB,2*N])
        inp = np.ones_like(inp)
        
        Ztest = np.random.rand(NZ,2*N,2*N)
        
        for k in range(1000):
            loss, r = lossF(inp,Ztest,target)
            
            if k % 100 == 0:
                print("Loss : {}\t{}".format(loss,r.mean()))
                
            DR = dR_dtheta(inp,Ztest)

            dl = 2*(r - target)
            
            J.set_value(J.get_value() - .001*np.tensordot(dl,DR[0],axes = [[0,1,2],[0,1,2]]).astype("float32"))
            D.set_value(D.get_value() - .001*np.tensordot(dl,DR[1],axes = [[0,1,2],[0,1,2]]).astype("float32"))
            S.set_value(S.get_value() - .001*np.tensordot(dl,DR[2],axes = [[0,1,2],[0,1,2]]).astype("float32"))

        exit()
        #I have verified that this does in fact seem to work

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

    DIS_red_r_true = SD.make_net(red_R_true,INSHAPE)
    DIS_red_r_fake = SD.make_net(red_R_fake,INSHAPE,params = lasagne.layers.get_all_layers(DIS_red_r_true))

    #get the outputs
    true_dis_out = lasagne.layers.get_output(DIS_red_r_true)
    fake_dis_out = lasagne.layers.get_output(DIS_red_r_fake)

    #make the loss functions
    true_loss_exp = -np.log(true_dis_out).mean() - np.log(1. - fake_dis_out).mean()#discriminator loss
    fake_loss_exp = -np.log(fake_dis_out).mean()#generative loss

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    D_updates = lasagne.updates.nesterov_momentum(true_loss_exp,lasagne.layers.get_all_params(DIS_red_r_true),.01)#discriminator training function

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

    lr = .01

    G_updates = lasagne.updates.adam([dLdJ_exp,dLdD_exp,dLdS_exp],[J,D,S],lr)
    G_train_func = theano.function([rvec,ivec,Z],fake_loss_exp,updates = G_updates,allow_input_downcast = True)
    D_train_func = theano.function([rvec,red_R_true],true_loss_exp,updates = D_updates,allow_input_downcast = True)

    #Now we set up values to use in testing.

    inp = BAND_IN

    def log(a,F = "./SSNGAN_log.log",PRINT = True):
        print(a)
        f = open(F,"a")
        f.write(str(a) + "\n")
        f.close()

    log("Gloss,Dloss")

    for k in range(iterations):
        rtest = np.zeros([NZ,NB,2*N])

        np.random.shuffle(data)

        true = data[:NZ]
        ztest = np.random.rand(NZ,2*N,2*N) 

        wtest = W(ztest)
        # Ztest = [[SSsolve.fixed_point(wtest[w],inp[i],r0 = rtest[w,i],
        #                               k = coe_value, n = exp_value)
        #           for i in range(len(inp))] for w in range(len(wtest))]

        rtest = np.array([[
            SSsolve.fixed_point(wtest[w],inp[i],r0 = rtest[w,i],
                                k = coe_value, n = exp_value).x
            for i in range(len(inp))] for w in range(len(wtest))])
        # import pdb
        # pdb.set_trace()
        # print(Ztest)

        Gloss = G_train_func(rtest,inp,ztest)
        Dloss = D_train_func(rtest,true)
        
        log("{},{}".format(Gloss,Dloss))

        if k%10 == 0:
            jj = J.get_value()
            dd = D.get_value()
            ss = S.get_value()
            
            allpar = np.reshape(np.concatenate([jj,dd,ss]),[-1]).tolist()

            string = "{},{},{},{},{},{},{},{},{},{},{},{}".format(allpar[0],
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

            log(string,F = "./parameters.log")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'datapath', default='training_data_TCs_Ne102.mat', nargs='?',
        help='Path to MATLAB data file (default: %(default)s)')
    parser.add_argument(
        '--iterations', default=1000, type=int,
        help='Number of iterations (default: %(default)s)')
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for random numbers (default: %(default)s)')
    ns = parser.parse_args()
    main(**vars(ns))
