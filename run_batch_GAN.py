import theano
import theano.tensor as T
import lasagne
import numpy as np

import discriminators.simple_discriminator as SD
import nips_madness.gradient_expressions.make_w_batch as make_w
import nips_madness.gradient_expressions.SS_grad as SSgrad
import nips_madness.ssnode as SSsolve

import time

def main():

    exp = theano.shared(2.2,name = "exp")
    coe = theano.shared(.04,name = "coe")

    J = theano.shared(np.array([[.1,1.],[1.,.1]]).astype("float32"),name = "j")
    D = theano.shared(np.array([[.1,1.],[1.,.1]]).astype("float32"),name = "d")
    S = theano.shared(np.array([[1.,1.],[1.,1.]]).astype("float32"),name = "s")

    Jp = T.exp(J)
    Dp = T.exp(D)
    Sp = T.exp(S)

    dJpJ = T.reshape(T.jacobian(T.reshape(Jp,[-1]),J),[2,2,2,2])
    dDpD = T.reshape(T.jacobian(T.reshape(Dp,[-1]),D),[2,2,2,2])
    dSpS = T.reshape(T.jacobian(T.reshape(Sp,[-1]),S),[2,2,2,2])
    
    n = theano.shared(50,name = "n_sites")
    nz = theano.shared(5,name = 'n_samples')
    nb = theano.shared(5,name = 'n_stim')

    X = theano.shared(np.linspace(-1,1,n.get_value()).astype("float32"),name = "positions")

    ##make our Z vector
    N = n.get_value()
    NZ = nz.get_value()
    NB = nb.get_value()
    m = 10
    ###

    #a mask to get from the rates to just the ones we are measuring
    M = theano.shared(np.array([[1 if k == j + (N/2) - (m/2) else 0 for k in range(2*N)] for j in range(m)]).astype("float32"),"mask")

    Z = T.tensor3("z","float32")

    #symbolic W
    ww = make_w.make_W_with_x(Z,Jp,Dp,Sp,n,X)

    #symbolic jacobians
    dwdj = T.reshape(T.jacobian(T.flatten(ww),J),[-1,2*n,2*n,2,2])
    dwdd = T.reshape(T.jacobian(T.flatten(ww),D),[-1,2*n,2*n,2,2])
    dwds = T.reshape(T.jacobian(T.flatten(ww),S),[-1,2*n,2*n,2,2])

#    dwdj = make_w.make_WJ_with_x(Z,Jp,Dp,Sp,n,X,dJpJ)
#    dwdd = make_w.make_WD_with_x(Z,Jp,Dp,Sp,n,X,dDpD)
#    dwds = make_w.make_WS_with_x(Z,Jp,Dp,Sp,n,X,dSpS)

    W = theano.function([Z],ww,allow_input_downcast = True,on_unused_input = "ignore")

    DWj = theano.function([Z],dwdj,allow_input_downcast = True,on_unused_input = "ignore")
    DWd = theano.function([Z],dwdd,allow_input_downcast = True,on_unused_input = "ignore")
    DWs = theano.function([Z],dwds,allow_input_downcast = True,on_unused_input = "ignore")

    Ztest = np.random.rand(NZ,2*N,2*N).astype("float32")

    print("DWj TEST: {}".format(np.abs(DWj(Ztest)).mean()))
    print("DWd TEST: {}".format(np.abs(DWd(Ztest)).mean()))
    print("DWs TEST: {}".format(np.abs(DWs(Ztest)).mean()))

    #now we need to get a function to generate dr/dth from dw/dth

    rvec = T.tensor3("rvec","float32")
    ivec = T.matrix("ivec","float32")

    dRdJ_exp = SSgrad.WRgrad_batch(rvec,ww,dwdj,ivec,exp,coe,NZ,NB,N)
    dRdD_exp = SSgrad.WRgrad_batch(rvec,ww,dwdd,ivec,exp,coe,NZ,NB,N)
    dRdS_exp = SSgrad.WRgrad_batch(rvec,ww,dwds,ivec,exp,coe,NZ,NB,N)

    prof = False

    dRdJ = theano.function([rvec,ivec,Z],dRdJ_exp,allow_input_downcast = True,profile = prof)
    dRdD = theano.function([rvec,ivec,Z],dRdD_exp,allow_input_downcast = True,profile = prof)
    dRdS = theano.function([rvec,ivec,Z],dRdS_exp,allow_input_downcast = True,profile = prof)

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


    timetest = True
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
     
    DRtest = True
    if DRtest:
    #I want to test this by adjusting the parameters to give some specified output
        
        def lossF(i,z,tar):
            WW = np.array([W(zz) for zz in z])
            
            r = np.array([[SSsolve.fixed_point(WW[n],i[m]).x for m in range(len(i))] for n in range(len(z))])
            
            return ((r - tar)**2).mean(),r
        
    ##If the target is tar, and the loss is sqred loss, then Dl/Dr = 2*(r - tar)
    ##so Dl/Dt = 2*(r - tar).Dr/Dth

        
        target = np.random.rand(1,1,2*N)
        target = np.ones_like(target)
        
        inp = np.random.normal(0,1,[1,2*N])
        inp = np.ones_like(inp)
        
        Ztest = np.random.rand(1,2*N,2*N)
        
        for k in range(1000):
            loss, r = lossF(inp,Ztest,target)
            
            if k % 100 == 0:
                print("Loss : {}\t{}".format(loss,r.mean()))
                
            DR = dR_dtheta(inp,Ztest)

            dl = 2*(r - target)
            
            J.set_value(J.get_value() + .001*np.tensordot(dl,DR[0],axes = [[0,1,2],[0,1,2]]).astype("float32"))
            D.set_value(D.get_value() + .001*np.tensordot(dl,DR[1],axes = [[0,1,2],[0,1,2]]).astype("float32"))
            S.set_value(S.get_value() + .001*np.tensordot(dl,DR[2],axes = [[0,1,2],[0,1,2]]).astype("float32"))

        exit()
        #I have verified that this does in fact seem to work

    #Now I need to make the GAN
    
    #I want to make a network that takes a tensor of shape [2N] and generates dl/dr

    red_R_true = T.vector("reduced rates","float32")

    red_R_fake = np.log(1 + T.tensordot(rvec,M,axes = [0,1]))
    
    get_reduced = theano.function([rvec],red_R_fake,allow_input_downcast = True)
    
    INSHAPE = [m]
 
    #I want to make a network that takes red_R and gives a scalar output

    DIS_red_r_true = SD.make_net(red_R_true,INSHAPE)
    DIS_red_r_fake = SD.make_net(red_R_fake,INSHAPE,params = lasagne.layers.get_all_layers(DIS_red_r_true))

    #get the outputs
    true_dis_out = lasagne.layers.get_output(DIS_red_r_true)
    fake_dis_out = lasagne.layers.get_output(DIS_red_r_fake)

    #make the loss functions
    true_loss_exp = -np.log(true_dis_out).mean() - np.log(1. - fake_dis_out).mean()
    fake_loss_exp = -np.log(fake_dis_out).mean()

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    true_dis_update = lasagne.updates.nesterov_momentum(true_loss_exp,lasagne.layers.get_all_params(DIS_red_r_true),.01)

    true_loss = theano.function([red_R_true,rvec],true_loss_exp,allow_input_downcast = True)
    fake_loss = theano.function([rvec],fake_loss_exp,allow_input_downcast = True)

    #to get the grads w.r.t. the generators parameters we need to do a jacobian 
    fake_dis_grad = T.jacobian(T.flatten(fake_loss_exp),rvec)
    fake_dis_grad = T.reshape(fake_dis_grad,[2*N])

    t_grad = theano.function([rvec],fake_dis_grad,allow_input_downcast = True)

    test_rates = np.random.normal(0,1,[2*N])

    dLdJ_exp = T.tensordot(fake_dis_grad,dRdJ_exp,axes = [0,0])
    dLdJ = theano.function([rvec,ivec,Z],dLdJ_exp,allow_input_downcast = True)

    inp = np.concatenate([np.ones(N),np.zeros(N)])
 
    ztest = np.random.rand(2*N,2*N)


    for k in range(1000):

        wtest = W(ztest)
        rtest = SSsolve.fixed_point(wtest,inp).x    

        dj = dLdJ(rtest,inp,ztest)
        J.set_value(J.get_value() - 1.*dj)        
    
        if k % 10 == 0:
            print(fake_loss(rtest))
            
if __name__ == "__main__":
    main()
