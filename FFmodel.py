import lasagne 
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD
import math 

niter = 10000

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

    pos = T.reshape(x,[-1,1,3])    
    dist_sq = ((pos - T.transpose(pos,[1,0,2]))**2).sum(axis = 2)
    input_activations = T.reshape((T.reshape(T.exp(-dist_sq/(2*(S**6))),[1,nx*ny*nz,nx*ny*nz])*T.reshape(inp,[ni,1,-1])).sum(axis = 2),[1,ni,1,-1])#[1,ni,1,nx*ny*nz]
    hidden_activations = (T.reshape((mW + sW*w),[nsam,1,nhid,-1])*input_activations).sum(axis = 3) + T.reshape((mB + sB*b),[nsam,1,nhid])#[nsam,ni,nhid]
    return hidden_activations

def run_GAN():

    nsam = theano.shared(10)
    nx = theano.shared(5)
    ny = theano.shared(5)
    nz = theano.shared(5)

    nhid = theano.shared(100)

    ni = theano.shared(10)
    
    ###
    NX = nx.get_value()
    NY = ny.get_value()
    NZ = nz.get_value()
    NSAM = nsam.get_value()
    NHID = nhid.get_value()
    NI = ni.get_value()
    ###

    S = theano.shared(np.float32(0),name = "RF_s")
    mW = theano.shared(np.float32(0),name = "mean_W")
    sW = theano.shared(np.float32(0),name = "s_W")
    mB = theano.shared(np.float32(0),name = "mean_b")
    sB = theano.shared(np.float32(0),name = "s_b")

    S_true = theano.shared(np.float32(-.5))
    mW_true = theano.shared(np.float32(.2))
    sW_true = theano.shared(np.float32(.5))
    mB_true = theano.shared(np.float32(-.3))
    sB_true = theano.shared(np.float32(.2))
      
    pos = theano.shared(np.array([[[[x,y,z] for z in range(nz.get_value())] for y in range(ny.get_value())] for x in range(nx.get_value())]).astype("float32"))
    
    def generate_samples():
        W = np.random.normal(mW.get_value(),np.exp(sW.get_value()),(NSAM,NHID,NX*NY*NZ)).astype("float32")
        B = np.random.normal(mB.get_value(),np.exp(sB.get_value()),(NSAM,NHID)).astype("float32")

        return W,B

    def generate_data():
        W = np.random.normal(mW_true.get_value(),np.exp(sW_true.get_value()),(NSAM,NHID,NX*NY*NZ)).astype("float32")
        B = np.random.normal(mB_true.get_value(),np.exp(sB_true.get_value()),(NSAM,NHID)).astype("float32")

        return W,B

    stimulus = T.tensor4("input","float32")
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

    loss = "CE"
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

    D_dat = SD.make_net(FOBS_dat,INSHAPE,loss,layers)
    D_sam = SD.make_net(FOBS_sam,INSHAPE,loss,layers,params = lasagne.layers.get_all_layers(D_dat))

    #get the outputs
    dat_out = lasagne.layers.get_output(D_dat)
    sam_out = lasagne.layers.get_output(D_sam)

    #make the loss functions
    SM = .9
    D_loss_exp = -SM*np.log(dat_out).mean() - (1. - SM)*np.log(1. - dat_out).mean() - np.log(1. - sam_out).mean()#discriminator loss
    G_loss_exp = -np.log(sam_out).mean()#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    D_updates = lasagne.updates.adam(D_loss_exp,lasagne.layers.get_all_params(D_dat), .001)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp,[S,mW,sW,mB,sB],.001)

    G_train_func = theano.function([weights,bias,weights_true,bias_true,stimulus],G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    D_train_func = theano.function([weights,bias,weights_true,bias_true,stimulus],D_loss_exp,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')
    
    #now define the actualy values and test

    STIM = np.random.randint(2,size = (NI,NX,NY,NZ))

    F = open("./FF_GAN_log.csv","w")
    F.write("dS\tdmW\tdsW\tdmB\tdsB\n")
    F.close()

    print("dS\tdmW\tdsW\tdmB\tdsB")
    for k in range(niter):
    
        SS = generate_samples()
        DD = generate_data()

        gloss = G_train_func(SS[0],SS[1],DD[0],DD[1],STIM)
        dloss = D_train_func(SS[0],SS[1],DD[0],DD[1],STIM)

        if k%1==0:
            ndb = 10
            ds = np.round((S.get_value() - S_true.get_value())**2,ndb)
            dmw = np.round((mW.get_value() - mW_true.get_value())**2,ndb)
            dsw = np.round((sW.get_value() - sW_true.get_value())**2,ndb)
            dmb = np.round((mB.get_value() - mB_true.get_value())**2,ndb)
            dsb = np.round((sB.get_value() - sB_true.get_value())**2,ndb)
            OUT="{}\t{}\t{}\t{}\t{}".format(ds,dmw,dsw,dmb,dsb)

            if k%100 == 0:
                print(OUT)
            F = open("./FF_GAN_log.csv","a")
            F.write(OUT + "\n")
            F.close()


def run_WGAN():

    nsam = theano.shared(100)
    nx = theano.shared(5)
    ny = theano.shared(5)
    nz = theano.shared(5)

    nhid = theano.shared(100)

    ni = theano.shared(10)
    
    ###
    NX = nx.get_value()
    NY = ny.get_value()
    NZ = nz.get_value()
    NSAM = nsam.get_value()
    NHID = nhid.get_value()
    NI = ni.get_value()
    ###

    S = theano.shared(np.float32(0),name = "RF_s")
    mW = theano.shared(np.float32(0),name = "mean_W")
    sW = theano.shared(np.float32(0),name = "s_W")
    mB = theano.shared(np.float32(0),name = "mean_b")
    sB = theano.shared(np.float32(0),name = "s_b")

    S_true = theano.shared(np.float32(-.5))
    mW_true = theano.shared(np.float32(.2))
    sW_true = theano.shared(np.float32(.5))
    mB_true = theano.shared(np.float32(-.3))
    sB_true = theano.shared(np.float32(.2))
      
    pos = theano.shared(np.array([[[[x,y,z] for z in range(nz.get_value())] for y in range(ny.get_value())] for x in range(nx.get_value())]).astype("float32"))
    
    def generate_samples():
        W = np.random.normal(mW.get_value(),np.exp(sW.get_value()),(NSAM,NHID,NX*NY*NZ)).astype("float32")
        B = np.random.normal(mB.get_value(),np.exp(sB.get_value()),(NSAM,NHID)).astype("float32")

        return W,B

    def generate_data():
        W = np.random.normal(mW_true.get_value(),np.exp(sW_true.get_value()),(NSAM,NHID,NX*NY*NZ)).astype("float32")
        B = np.random.normal(mB_true.get_value(),np.exp(sB_true.get_value()),(NSAM,NHID)).astype("float32")

        return W,B

    stimulus = T.tensor4("input","float32")
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

    loss = "WGAN"
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

    D_D_input = T.tensor3("DDinput","float32")
    D_G_input = T.tensor3("DGinput","float32")

    get_DD_input = theano.function([weights_true,bias_true,stimulus],FOBS_dat,allow_input_downcast = True)
    get_DG_input = theano.function([weights,bias,stimulus],FOBS_sam,allow_input_downcast = True)

    D_dat = SD.make_net(D_D_input,INSHAPE,loss,layers)
    D_sam = SD.make_net(D_G_input,INSHAPE,loss,layers,params = lasagne.layers.get_all_layers(D_dat))
    G_sam = SD.make_net(FOBS_sam,INSHAPE,loss,layers,params = lasagne.layers.get_all_layers(D_dat))

    #get the outputs
    D_dat_out = lasagne.layers.get_output(D_dat)
    D_sam_out = lasagne.layers.get_output(D_sam)
    G_sam_out = lasagne.layers.get_output(G_sam)

    #make the loss functions
    
    D_grad_input = T.tensor3("D_sam","float32")

    D_grad_net = SD.make_net(D_grad_input,INSHAPE,loss,layers,params = lasagne.layers.get_all_layers(D_dat))

    D_grad_out = lasagne.layers.get_output(D_grad_net)

    Dgrad = T.sqrt((T.jacobian(T.reshape(D_grad_out,[-1]),D_grad_input)**2).sum(axis = [1,2,3]))
    Dgrad_penalty = ((Dgrad - 1.)**2).mean()

    SM = .9
    lam = .00001

    D_loss_exp = D_sam_out.mean() - D_dat_out.mean() + lam*Dgrad_penalty#discriminator loss
    G_loss_exp = -G_sam_out.mean()#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .001
    D_updates = lasagne.updates.adam(D_loss_exp,lasagne.layers.get_all_params(D_dat), .001)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp,[S,mW,sW,mB,sB],.001)

    G_train_func = theano.function([weights,bias,stimulus],G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    D_train_func = theano.function([D_D_input,D_G_input,D_grad_input],Dgrad_penalty,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')
    
    #now define the actualy values and test

    STIM = np.random.randint

    def make_inp(p):
        PP = pos.get_value()

        PNT = np.reshape(p,[1,1,1,3])

        return np.exp(-((PP - PNT)**2).sum(axis = 3)/(2*(.5)**6))

    locs = np.array([[.5,.5 + .25*np.cos((2*math.pi*k)/NI),.5 + .25*np.sin((2*math.pi*k)/NI)] for k in range(NI)])

    STIM = np.array([make_inp(p) for p in locs])

    NDstep = 5

    F = open("./FF_log.csv","w")
    F.write("dS\tdmW\tdsW\tdmB\tdsB\n")
    F.close()

    print("dS\tdmW\tdsW\tdmB\tdsB")

    for k in range(niter):
    
        for dstep in range(NDstep):
            #get the samples for training
            SS = generate_samples()
            DD = generate_data()

            #get the data inputs
            data = get_DD_input(DD[0],DD[1],STIM)
            samples = get_DG_input(SS[0],SS[1],STIM)

            SHAPE = samples.shape

            #generate random numbers for interpolation
            EE = np.random.rand(SHAPE[0],SHAPE[1],SHAPE[2])
            gradpoints = EE*data + (1. - EE)*samples            

            #compute the update
            dloss = D_train_func(data,samples,gradpoints)

        SS = generate_samples()  
        gloss = G_train_func(SS[0],SS[1],STIM)
        
        if k%1==0:
            ndm = 10
            ds = np.round((S.get_value() - S_true.get_value())**2,ndm)
            dmw = np.round((mW.get_value() - mW_true.get_value())**2,ndm)
            dsw = np.round((sW.get_value() - sW_true.get_value())**2,ndm)
            dmb = np.round((mB.get_value() - mB_true.get_value())**2,ndm)
            dsb = np.round((sB.get_value() - sB_true.get_value())**2,ndm)
            OUT = "{}\t{}\t{}\t{}\t{}".format(ds,dmw,dsw,dmb,dsb)

            print(OUT)

            F = open("./FF_log.csv","a")
            F.write(OUT + "\n")
            F.close()

if __name__ == "__main__":
    run_GAN()
