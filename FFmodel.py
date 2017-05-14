import lasagne 
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD
import math 
import sys 

niter = 20000
np.random.seed(0)

#tag = "debug"
tag = ""

S = theano.shared(np.float32(-1),name = "RF_s")
mW = theano.shared(np.float32(.5),name = "mean_W")
sW = theano.shared(np.float32(-1),name = "s_W")
mB = theano.shared(np.float32(-.5),name = "mean_b")
sB = theano.shared(np.float32(.5),name = "s_b")

S_true = theano.shared(np.float32(-2.))
mW_true = theano.shared(np.float32(0))
sW_true = theano.shared(np.float32(0))
mB_true = theano.shared(np.float32(0))
sB_true = theano.shared(np.float32(0))


def get_FF_output(S,mW,sW,mB,sB,w,b,x,inp,nsam,nx,ny,nz,nhid,ni):

    """

    S - width of RF - scalar
    mW - mean of the weights - scalar
    sW - std. of weights - scalar
    mB - mean of thresholds - scalar
    sB - std. of thresholds - scalar
    w - w samples - [nsam,nhid,nx*ny*nz]
    b - b samples - [nsam,nhid]
    x - coordinate positions - [nx,ny,nz,3]
    inp - inputs - [ni,nx,ny,nz]

    """

    pos = T.reshape(x,[1,-1,3])    
    stim = T.reshape(inp,[ni,1,3])
    dist_sq = ((pos - stim)**2).sum(axis = 2)#[ni,nx*ny*nz]
    input_activations = T.reshape(T.exp(-dist_sq/(2*(S**6))),[1,ni,1,-1])#[1,ni,nx*ny*nz]
    hidden_activations = (T.reshape((mW + sW*w),[nsam,1,nhid,-1])*input_activations).sum(axis = 3) + T.reshape((mB + sB*b),[nsam,1,nhid])#[nsam,ni,nhid]
    return hidden_activations

def run_GAN(mode):

    box_width = 10

    nsam = theano.shared(10)
    nx = theano.shared(box_width)
    ny = theano.shared(box_width)
    nz = theano.shared(box_width)

    nhid = theano.shared(100)

    ni = theano.shared(box_width)
    
    ###
    NX = nx.get_value()
    NY = ny.get_value()
    NZ = nz.get_value()
    NSAM = nsam.get_value()
    NHID = nhid.get_value()
    NI = ni.get_value()
    ###

    XX = np.linspace(-1,1,NX).astype("float32")
    YY = np.linspace(-1,1,NY).astype("float32")
    ZZ = np.linspace(-1,1,NZ).astype("float32")
      
    pos = theano.shared(np.array([[[[x,y,z] for z in ZZ] for y in YY] for x in XX]).astype("float32"))

    #generate the right shape 
    def generate_samples():
        W = np.random.normal(0,1,(NSAM,NHID,NX*NY*NZ)).astype("float32")
        B = np.random.normal(0,1,(NSAM,NHID)).astype("float32")

        return W,B

    stimulus = T.matrix("input","float32")
    weights = T.tensor3("weights","float32")
    bias = T.matrix("bias","float32")

    weights_true = T.tensor3("weights","float32")
    bias_true = T.matrix("bias","float32")
    
    FFout_sam = get_FF_output(T.exp(S),mW,T.exp(sW),mB,T.exp(sB),weights,bias,pos,stimulus,nsam,nx,ny,nz,nhid,ni)
    output_sam = theano.function([weights,bias,stimulus],FFout_sam,allow_input_downcast = True)

    FFout_dat = get_FF_output(T.exp(S_true),mW_true,T.exp(sW_true),mB_true,T.exp(sB_true),weights_true,bias_true,pos,stimulus,nsam,nx,ny,nz,nhid,ni)
    output_dat = theano.function([weights_true,bias_true,stimulus],FFout_dat,allow_input_downcast = True)
    
    #done defining the output function

    NOBS = 10
    INSHAPE = (NSAM,NI,NOBS)

    layers = [128,128]
    
    def make_mask():
        sel = np.random.choice(np.arange(NHID),NOBS)
        out = np.zeros((NOBS,NHID))
        for k in range(len(sel)):
            out[k,sel[k]] = 1

        return np.reshape(out,[NOBS,NHID]).astype("float32")

    mask = theano.shared(make_mask(),name = "mask")

    FOBS_dat = (T.reshape(mask,[1,1,NOBS,nhid])*T.reshape(FFout_dat,[nsam,ni,1,nhid])).sum(axis = 3)
    FOBS_sam = (T.reshape(mask,[1,1,NOBS,nhid])*T.reshape(FFout_sam,[nsam,ni,1,nhid])).sum(axis = 3)

    get_DD_input = theano.function([weights_true,bias_true,stimulus],FOBS_dat,allow_input_downcast = True)
    get_DG_input = theano.function([weights,bias,stimulus],FOBS_sam,allow_input_downcast = True)

    #I need a function that returns training functions
    D_train, G_train = make_train_funcs(FOBS_sam,[S,mW,sW,mB,sB],[weights,bias,stimulus],[128,128],(NSAM,NI,NOBS),mode)
    
    #now define the actual values and test

    STIM = np.array([[0,.25*np.cos((2*math.pi*k)/NI),.25 * np.sin((2*math.pi*k)/NI)] for k in range(NI)])

    train(D_train,G_train,get_DD_input,get_DG_input,generate_samples,STIM,mode = mode)

def train(D_step,G_step,D_in,G_in,F_gen,STIM,mode = "WGAN",tag = "test"):
    
    if mode =="WGAN":
        train_wgan(D_step,G_step,D_in,G_in,F_gen,STIM,tag)
    elif mode =="GAN":
        train_gan(D_step,G_step,D_in,G_in,F_gen,STIM,tag)

def train_wgan(D_step,G_step,D_in,G_in,F_gen,STIM,tag = "test"):
    
    NDstep = 5
    
    F = open("./FF_logs/FF_WGAN_log_"+tag+".csv","w")
    F.write("dS\tdmW\tdsW\tdmB\tdsB\n")
    F.close()

    F = open("./FF_logs/FF_WGAN_losslog_"+tag+".csv","w")
    F.write("gloss,dloss\n")
    F.close()

    print("dS\tdmW\tdsW\tdmB\tdsB")

    gloss = 10.
    dloss = 10.

    for k in range(niter):
    
        for dstep in range(NDstep):
            #get the samples for training
            SS = F_gen()
            DD = F_gen()

            #get the data inputs
            data = D_in(DD[0],DD[1],STIM)
            samples = G_in(SS[0],SS[1],STIM)

            SHAPE = samples.shape

            #generate random numbers for interpolation
            EE = np.random.rand(SHAPE[0],SHAPE[1],SHAPE[2])
            gradpoints = EE*data + (1. - EE)*samples            

            #compute the update
            dloss = D_step(data,samples,gradpoints)

            F = open("./FF_logs/FF_WGAN_losslog_"+tag+".csv","a")
            F.write("{},{}\n".format(gloss,dloss))
            F.close()


        SS = F_gen()
        gloss = G_step(SS[0],SS[1],STIM)

        F = open("./FF_logs/FF_WGAN_losslog_"+tag+".csv","a")
        F.write("{},{}\n".format(gloss,dloss))
        F.close()
        
        if k%1==0:
            ndm = 10
            ds = np.round((S.get_value() - S_true.get_value())**2,ndm)
            dmw = np.round((mW.get_value() - mW_true.get_value())**2,ndm)
            dsw = np.round((sW.get_value() - sW_true.get_value())**2,ndm)
            dmb = np.round((mB.get_value() - mB_true.get_value())**2,ndm)
            dsb = np.round((sB.get_value() - sB_true.get_value())**2,ndm)
            OUT = "{}\t{}\t{}\t{}\t{}".format(ds,dmw,dsw,dmb,dsb)

            print(OUT)

            F = open("./FF_logs/FF_WGAN_log_"+tag+".csv","a")
            F.write(OUT + "\n")
            F.close()

    
def train_gan(D_step,G_step,D_in,G_in,F_gen,STIM,tag = "test"):
        
    F = open("./FF_logs/FF_GAN_log_"+tag+".csv","w")
    F.write("dS\tdmW\tdsW\tdmB\tdsB\n")
    F.close()

    F = open("./FF_logs/FF_GAN_losslog_"+tag+".csv","w")
    F.write("gloss,dloss\n")
    F.close()

    print("dS\tdmW\tdsW\tdmB\tdsB")

    gloss = 10.
    dloss = 10.

    for k in range(niter):
    
        SS = F_gen()
        DD = F_gen()
        
        #get the disc inputs
        data = D_in(DD[0],DD[1],STIM)
        samples = G_in(SS[0],SS[1],STIM)

        #compute the update
        dloss = D_step(data,samples)
        gloss = G_step(SS[0],SS[1],STIM)

        F = open("./FF_logs/FF_GAN_losslog_"+tag+".csv","a")
        F.write("{},{}\n".format(gloss,dloss))
        F.close()
        
        if k%1==0:
            ndm = 10
            ds = np.round((S.get_value() - S_true.get_value())**2,ndm)
            dmw = np.round((mW.get_value() - mW_true.get_value())**2,ndm)
            dsw = np.round((sW.get_value() - sW_true.get_value())**2,ndm)
            dmb = np.round((mB.get_value() - mB_true.get_value())**2,ndm)
            dsb = np.round((sB.get_value() - sB_true.get_value())**2,ndm)
            OUT = "{}\t{}\t{}\t{}\t{}".format(ds,dmw,dsw,dmb,dsb)

            print(OUT)

            F = open("./FF_logs/FF_GAN_log_"+tag+".csv","a")
            F.write(OUT + "\n")
            F.close()

def make_train_funcs(generator, Gparams, Ginputs, layers, INSHAPE,mode = "WGAN"):

    if mode == "WGAN":
        return make_WGAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE)
    elif mode =="GAN":
        return make_GAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE)

def make_WGAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE):
    D_D_input = T.tensor3("DDinput","float32")
    D_G_input = T.tensor3("DGinput","float32")    
    
    D_dat = SD.make_net(D_D_input,INSHAPE,"WGAN",layers)
    D_sam = SD.make_net(D_G_input,INSHAPE,"WGAN",layers,params = lasagne.layers.get_all_layers(D_dat))
    G_sam = SD.make_net(generator,INSHAPE,"WGAN",layers,params = lasagne.layers.get_all_layers(D_dat))
    
    #get the outputs
    D_dat_out = lasagne.layers.get_output(D_dat)
    D_sam_out = lasagne.layers.get_output(D_sam)
    G_sam_out = lasagne.layers.get_output(G_sam)
    
    #make the loss functions
    
    D_grad_input = T.tensor3("D_sam","float32")
    
    D_grad_net = SD.make_net(D_grad_input,INSHAPE,"WGAN",layers,params = lasagne.layers.get_all_layers(D_dat))

    D_grad_out = lasagne.layers.get_output(D_grad_net)

    Dgrad = T.sqrt((T.jacobian(T.reshape(D_grad_out,[-1]),D_grad_input)**2).sum(axis = [1,2,3]))
    Dgrad_penalty = ((Dgrad - 1.)**2).mean()

    lam = 10.

    D_loss_exp = D_sam_out.mean() - D_dat_out.mean() + lam*Dgrad_penalty#discriminator loss
    G_loss_exp = - G_sam_out.mean()#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .0001
    
    b1 = .5
    b2 = .9

    D_updates = lasagne.updates.adam(D_loss_exp, lasagne.layers.get_all_params(D_dat), lr,beta1 = b1,beta2 = b2)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp, Gparams, lr,beta1 = b1,beta2 = b2)

    G_train_func = theano.function(Ginputs,G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    D_train_func = theano.function([D_D_input,D_G_input,D_grad_input],Dgrad_penalty,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    return D_train_func, G_train_func

def make_GAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE):
    D_D_input = T.tensor3("DDinput","float32")
    D_G_input = T.tensor3("DGinput","float32")    
    
    D_dat = SD.make_net(D_D_input,INSHAPE,"CE",layers)
    D_sam = SD.make_net(D_G_input,INSHAPE,"CE",layers,params = lasagne.layers.get_all_layers(D_dat))
    G_sam = SD.make_net(generator,INSHAPE,"CE",layers,params = lasagne.layers.get_all_layers(D_dat))
    
    #get the outputs
    D_dat_out = lasagne.layers.get_output(D_dat)
    D_sam_out = lasagne.layers.get_output(D_sam)
    G_sam_out = lasagne.layers.get_output(G_sam)
    
    #make the loss functions
    D_loss_exp = - T.log(D_dat_out.mean()) - T.log(1. - D_sam_out.mean())#discriminator loss
    G_loss_exp = - T.log(G_sam_out.mean())#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .0001
    
    b1 = .5
    b2 = .9

    D_updates = lasagne.updates.adam(D_loss_exp, lasagne.layers.get_all_params(D_dat), lr,beta1 = b1,beta2 = b2)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp, Gparams, lr,beta1 = b1,beta2 = b2)

    G_train_func = theano.function(Ginputs,G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    D_train_func = theano.function([D_D_input,D_G_input],D_loss_exp,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    return D_train_func, G_train_func

if __name__ == "__main__":
    run_GAN(sys.argv[1])
