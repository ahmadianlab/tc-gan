import lasagne 
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD

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

if __name__ == "__main__":

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

    S = theano.shared(np.float32(11.),name = "RF_s")
    mW = theano.shared(np.float32(1.1),name = "mean_W")
    sW = theano.shared(np.float32(.55),name = "s_W")
    mB = theano.shared(np.float32(-2.2),name = "mean_b")
    sB = theano.shared(np.float32(3.3),name = "s_b")

    S_true = theano.shared(np.float32(10.))
    mW_true = theano.shared(np.float32(1.))
    sW_true = theano.shared(np.float32(.5))
    mB_true = theano.shared(np.float32(-2.))
    sB_true = theano.shared(np.float32(3.))
      
    pos = theano.shared(np.array([[[[x,y,z] for z in range(nz.get_value())] for y in range(ny.get_value())] for x in range(nx.get_value())]).astype("float32"))
    
    def generate_samples():
        W = np.random.normal(mW.get_value(),sW.get_value(),(NSAM,NHID,NX*NY*NZ)).astype("float32")
        B = np.random.normal(mB.get_value(),sB.get_value(),(NSAM,NHID)).astype("float32")

        return W,B

    def generate_data():
        W = np.random.normal(mW_true.get_value(),sW_true.get_value(),(NSAM,NHID,NX*NY*NZ)).astype("float32")
        B = np.random.normal(mB_true.get_value(),sB_true.get_value(),(NSAM,NHID)).astype("float32")

        return W,B

    stimulus = T.tensor4("input","float32")
    weights = T.tensor3("weights","float32")
    bias = T.matrix("bias","float32")

    weights_true = T.tensor3("weights","float32")
    bias_true = T.matrix("bias","float32")
    
    FFout_sam = get_FF_output(S,mW,sW,mB,sB,weights,bias,pos,stimulus,nsam,nx,ny,nz,nhid,ni)
    output_sam = theano.function([weights,bias,stimulus],FFout_sam,allow_input_downcast = True)

    FFout_dat = get_FF_output(S_true,mW_true,sW_true,mB_true,sB_true,weights_true,bias_true,pos,stimulus,nsam,nx,ny,nz,nhid,ni)
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

    print("dS\tdmW\tdsW\tdmB\tdsB")
    for k in range(10000):
    
        SS = generate_samples()
        DD = generate_data()

        gloss = G_train_func(SS[0],SS[1],DD[0],DD[1],STIM)
        dloss = D_train_func(SS[0],SS[1],DD[0],DD[1],STIM)

        if k%100==0:
            ds = np.round((S.get_value() - S_true.get_value())**2,3)
            dmw = np.round((mW.get_value() - mW_true.get_value())**2,3)
            dsw = np.round((sW.get_value() - sW_true.get_value())**2,3)
            dmb = np.round((mB.get_value() - mB_true.get_value())**2,3)
            dsb = np.round((sB.get_value() - sB_true.get_value())**2,3)
            print("{}\t{}\t{}\t{}\t{}".format(ds,dmw,dsw,dmb,dsb))
