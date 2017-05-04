import theano
import theano.tensor as T
import lasagne
import numpy as np

import discriminators.simple_discriminator as SD
import nips_madness.gradient_expressions.make_w as make_w
import nips_madness.gradient_expressions.SS_grad as SSgrad
import nips_madness.ssnode as SSsolve

import time

def main():

    exp = theano.shared(2.2,name = "exp")
    coe = theano.shared(.04,name = "coe")

    J = theano.shared(np.array([[.1,1.],[1.,.1]]),name = "j")
    D = theano.shared(np.array([[.1,1.],[1.,.1]]),name = "d")
    S = theano.shared(np.array([[1.,1.],[1.,1.]]),name = "s")

    n = theano.shared(3,name = "n_sites")
    nz = theano.shared(1,name = 'n_samples')
    nb = theano.shared(1,name = 'n_stim')

    X = theano.shared(np.linspace(-1,1,n.get_value()),name = "positions")

    ##make our Z vector
    N = n.get_value()
    NZ = nz.get_value()
    NB = nb.get_value()
    m = 2
    ###    

    #a mask to get from the rates to just the ones we are measuring
    M = theano.shared(np.array([[1 if k == j + (N/2) - (m/2) else 0 for k in range(2*N)] for j in range(m)]),"mask")

    Z = T.matrix("z","float32")

    #symbolic W
    ww = make_w.make_W_with_x(Z,J,D,S,n,X)

    #symbolic jacobians
    dwdj = T.reshape(T.jacobian(T.flatten(ww),J),[2*n,2*n,2,2])
    dwdd = T.reshape(T.jacobian(T.flatten(ww),D),[2*n,2*n,2,2])
    dwds = T.reshape(T.jacobian(T.flatten(ww),S),[2*n,2*n,2,2])

    W = theano.function([Z],ww,allow_input_downcast = True)

    DWj = theano.function([Z],dwdj,allow_input_downcast = True)
    DWd = theano.function([Z],dwdd,allow_input_downcast = True)
    DWs = theano.function([Z],dwds,allow_input_downcast = True)

    print("DWj TEST: {}".format(np.abs(DWj(np.random.rand(2*N,2*N))).mean()))
    print("DWd TEST: {}".format(np.abs(DWd(np.random.rand(2*N,2*N))).mean()))
    print("DWs TEST: {}".format(np.abs(DWs(np.random.rand(2*N,2*N))).mean()))

    #now we need to get a function to generate dr/dth from dw/dth

    rvec = T.vector("rvec","float32")
    ivec = T.vector("ivec","float32")

    dRdJ_exp = SSgrad.WRgrad(rvec,ww,dwdj,ivec,exp,coe)
    dRdD_exp = SSgrad.WRgrad(rvec,ww,dwdd,ivec,exp,coe)
    dRdS_exp = SSgrad.WRgrad(rvec,ww,dwds,ivec,exp,coe)

    dRdJ = theano.function([rvec,ivec,Z],dRdJ_exp,allow_input_downcast = True)
    dRdD = theano.function([rvec,ivec,Z],dRdD_exp,allow_input_downcast = True)
    dRdS = theano.function([rvec,ivec,Z],dRdS_exp,allow_input_downcast = True)

    testDW = False
    if testDW:
        testz = np.random.rand(2*N,2*N)
        testI = np.random.normal(0,10,2*N)
        
        wtest = W(testz)
        ssR = SSsolve.fixed_point(wtest,testI).x
        print(ssR.mean())
        print(wtest.mean())
        print(DWj(testz).mean())
        
        
        print("Start DWtest")
        for k in range(10000):
            wtest = W(testz)
            
            dd = DWj(testz)
            dj = 2*np.tensordot(wtest,dd,axes = [[0,1],[0,1]])
            
            jnew = J.get_value() - .01*dj
            
            J.set_value(jnew)
            
            if k % 1000 == 0:
                print((wtest*wtest).mean())

        exit()
        #running this will verify that the W gradient is working (by adjusting parameters to minimize the mean sqaured W entry)

    print("defined all the functions")

    def dR_dtheta(inp,ZZ):
        '''

        I need a function which takes 

          inp - [nb,2N]
          Z - [nz,2N,2N]

        and generates dr/dth

        '''

        if len(ZZ.shape) != 3:
            print("Z must have three dimentions - [nz,2N,2N]")

        if len(inp.shape) != 2:
            print("inp must have two dimentions - [nb,2N]")

        #[nz,2n,2n]
        WW = np.array([W(z) for z in ZZ])

#        print("got W")

        #[nz,nb,2N]
        RR = np.array([[SSsolve.fixed_point(WW[z],inp[i]).x for i in range(len(inp))] for z in range(len(ZZ))])

#        print("got steady state {}".format(RR.mean()))

        DRTj = [[dRdJ(RR[z,b],inp[b],ZZ[z]) for b in range(len(inp))] for z in range(len(ZZ))]
        DRTd = [[dRdD(RR[z,b],inp[b],ZZ[z]) for b in range(len(inp))] for z in range(len(ZZ))]
        DRTs = [[dRdS(RR[z,b],inp[b],ZZ[z]) for b in range(len(inp))] for z in range(len(ZZ))]
        
        DR = np.array([DRTj,DRTd,DRTs])

        return DR


    zz = np.random.rand(NZ,2*N,2*N)
    II = np.random.normal(0,1,[NB,2*N])

    print("Computing gradients")

    T_temp = time.time()

    DR = dR_dtheta(II,zz)
    print(DR.mean())

    print("A single datapoint took {} seconds to get dRdTH".format(time.time() - T_temp))
    print("An 8*5 batch would take {} seconds to get dRdTH".format(8*5*(time.time() - T_temp)))

    print(DR.shape)

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
    Ztest = np.random.rand(1,2*N,2*N)

    for k in range(10000):
        loss, r = lossF(inp,Ztest,target)

        if k % 1000 == 0:
            print("Loss : {}".format(loss))

        DR = dR_dtheta(inp,Ztest)

        dl = 2*(r - target)

        J.set_value(J.get_value() - 1*np.tensordot(dl,DR[0],axes = [[0,1,2],[0,1,2]]))
        D.set_value(D.get_value() - 1*np.tensordot(dl,DR[1],axes = [[0,1,2],[0,1,2]]))
        S.set_value(S.get_value() - 1*np.tensordot(dl,DR[2],axes = [[0,1,2],[0,1,2]]))

    exit()

    #I want to make a network that takes a tensor of shape [nz,nb,2N] and generates dl/dr

    red_R_true = T.tensor3("reduced rates","float32")

    rates = T.tensor3("rates","float32")  
    red_R_fake = T.tensordot(rates,M,axes = [2,1])
   
    nb = 20
 
    INSHAPE = (NZ,nb,m)
    
 
    #I want to make a network that takes red_R and gives a scalar output

    DIS_red_r_true = SD.make_net(red_R_true,INSHAPE)
    DIS_red_r_fake = SD.make_net(red_R_fake,INSHAPE)

    #get the outputs
    true_dis_out = lasagne.layers.get_output(DIS_red_r_true)
    fake_dis_out = lasagne.layers.get_output(DIS_red_r_fake)

    #tie the parameters
    Dparams = lasagne.layers.get_all_params(DIS_red_r_true)
    Dparams = lasagne.layers.get_all_params(DIS_red_r_fake)

    #make the loss functions
    true_loss = -np.log(true_dis_out).mean() - np.log(1. - fake_dis_out).mean()
    fake_loss = -np.log(fake_dis_out).mean()

    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    true_dis_update = lasagne.updates.nesterov_momentum(true_loss,lasagne.layers.get_all_params(DIS_red_r_true),.01)

    #to get the grads w.r.t. the generators parameters we need to do a jacobian 
    fake_dis_grad_temp = T.jacobian(T.flatten(fake_dis_out),rates)
    fake_dis_grad = T.reshape(fake_dis_grad_temp,[nz,nz,nb,2*N])

    t_grad = theano.function([rates],fake_dis_grad,allow_input_downcast = True)


    test_rates = np.random.normal(0,1,[NZ,nb,2*N])

    oo = t_grad(test_rates)

    print(oo.shape)

if __name__ == "__main__":
    main()
