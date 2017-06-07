import lasagne 
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD
import math 
import sys 
import time
import FF_functions.lalazar_func as IO_func

DATA = True

def read_dat(F):
    f = open(F,"r")

    out = []
    for l in f:
        temp = l.split(",")
        temp[-1] = temp[-1][:-1]

        out.append([float(t) for t in temp])
    f.close()
        
    return np.array(out)

def read_log(F):
    f = open(F,"r")

    out = []
    n = 0
    for l in f:
        temp = l.split(",")
        temp[-1] = temp[-1][:-1]

        if n != 1:
            out.append([float(t) for t in temp])
        else:
            out.append([t for t in temp])
        
        n += 1

    f.close()
        
    return np.array(out)

niter = 100000

np.random.seed(1)

tag = "pruned"

start_params = [np.log(2.),np.log(.25),np.log(.01),10.,np.log(10.),np.log(1.)]

#import the data
curves = read_dat("lalazar_data/TuningCurvesFull_Pronation.dat")

#the initial parameters
RF_low = theano.shared(np.float32(start_params[0]),name = "RF_s")
RF_del = theano.shared(np.float32(start_params[1]),name = "mean_W")
THR = theano.shared(np.float32(start_params[3]),name = "s_W")
THR_del = theano.shared(np.float32(start_params[4]),name = "s_W")
Js = theano.shared(np.float32(start_params[2]),name = "mean_b")
As = theano.shared(np.float32(start_params[5]),name = "mean_b")

#parameters:
# - RF_low
# - delta_RF
# - threshold
# - weight_scale (why not?)

#random saples:
# - RF width (uniform [0,1]) - [nsam,nx,ny,nz]
# - FF connections (discreete {0,1}) - [nsam,nhid,nx*ny*nz]
# - FF strengths (uniform [0,1]) - [nsam,nhid,nx*ny*nz]

def run_GAN(mode = "WGAN"):

    box_width = 100

    dx = 1./box_width

    nsam = theano.shared(20)
    nx = theano.shared(box_width)
    ny = theano.shared(box_width)
    nz = theano.shared(box_width)

    NX = nx.get_value()
    NY = ny.get_value()
    NZ = nz.get_value()

    nhid = theano.shared(1)

    XX = np.linspace(-3,3,NX).astype("float32")
    YY = np.linspace(-3,3,NY).astype("float32")
    ZZ = np.linspace(-3,3,NZ).astype("float32")
      
    pos = np.reshape(np.array([[[[x,y,z] for z in ZZ] for y in YY] for x in XX]).astype("float32"),[-1,3])

    STIM = np.array([[x,y,z] for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1]])

    ni = theano.shared(len(STIM))
    
    ###
    NSAM = nsam.get_value()
    NHID = nhid.get_value()
    NI = ni.get_value()

    NFF = int((box_width**3)/100)
    ###

    #generate the right shape 
    def generate_samples():
        wid = np.random.rand(NSAM,NFF)

        RF_p = np.array([pos[np.random.choice(np.arange(len(pos)),NFF)] for k in range(NSAM)])

        FF_s = np.random.rand(NSAM,NHID,NFF)
        TH_s = np.random.uniform(-1,1,[NSAM,NHID])

        return FF_s,wid,TH_s,RF_p

    stimulus = T.matrix("input","float32")

    feedforward_pos = T.tensor3("connts","float32")
    feedforward_strn = T.tensor3("streng","float32")
    feedforward_thrs = T.matrix("streng","float32")
    receptive_widths = T.matrix("widths","float32")


    PARAM = [RF_low,RF_del,THR,THR_del,Js,As]

    FFout_sam = IO_func.get_FF_output_pruned(T.exp(RF_low),T.exp(RF_del),THR,T.exp(THR_del),T.exp(Js),T.exp(As),receptive_widths,feedforward_strn,feedforward_thrs,feedforward_pos,stimulus,nsam,nx,ny,nz,nhid,ni,dx)

    GINP = [feedforward_strn,receptive_widths,feedforward_thrs,feedforward_pos,stimulus]

    output_sam = theano.function(GINP,FFout_sam,allow_input_downcast = True,on_unused_input = 'ignore')

    #done defining the output function

    NOBS = 1
    INSHAPE = (NSAM,NI,NOBS)

    layers = [128,128]
    
    def make_mask():
        sel = np.random.choice(np.arange(NHID),NOBS)
        out = np.zeros((NOBS,NHID))
        for k in range(len(sel)):
            out[k,sel[k]] = 1

        return np.reshape(out,[NOBS,NHID]).astype("float32")

    mask = theano.shared(make_mask(),name = "mask")

    FOBS_sam = (T.reshape(mask,[1,1,NOBS,nhid])*T.reshape(FFout_sam,[nsam,ni,1,nhid])).sum(axis = 3)

    get_DG_input = theano.function(GINP,FOBS_sam,allow_input_downcast = True)

    #I need a function that returns training functions
    D_train, G_train, DOUT = make_train_funcs(FOBS_sam,PARAM,GINP,[128,128],(NSAM,NI,NOBS),mode)

    print tag

    #now define the actual values and test
    train(D_train,G_train,get_DG_input,generate_samples,STIM,(NSAM,NI,NOBS),mode,tag + str(box_width),DATA)

def train(D_step,G_step,G_in,F_gen,STIM,INSHAPE,mode,tag,data,NDstep = 5):
    
    if mode =="WGAN":
        train_wgan(D_step,G_step,G_in,F_gen,STIM,INSHAPE,tag,data,NDstep)
    elif mode =="GAN":
        train_gan(D_step,G_step,G_in,F_gen,STIM,INSHAPE,tag,data)

def train_wgan(D_step,G_step,G_in,F_gen,STIM,INSHAPE,tag,use_data,NDstep = 5):
   
    print("Trainign WGAN")
 
    LOG = "./FF_logs/FF_log_"+tag+".csv"
    LOSS = "./FF_logs/FF_losslog_"+tag+".csv"
 
    F = open(LOG,"w")
    F.write("RF\tRFd\tJ\tth\tth_d\n")
    F.close()

    F = open(LOSS,"w")
    F.write("gloss,dloss\n")
    F.close()

    print("{}\t{}\t{}\t{}\t{}\t{}\n".format(RF_low.get_value(),
                                            RF_del.get_value(),
                                            Js.get_value(),
                                            THR.get_value(),
                                            THR_del.get_value(),
                                            As.get_value()))
    
    print("dRF\tdRFd\tdJ\tdth\tth_d\tAs")

    gloss = 10.
    dloss = 10.

    for k in range(niter):
    
        for dstep in range(NDstep):
            #get the samples for training
            SS = F_gen()
            DD = F_gen()

            #get the data inputs

            data = np.reshape(curves[np.random.choice(np.arange(len(curves)),INSHAPE[0])],INSHAPE)

            samples = G_in(SS[0],SS[1],SS[2],SS[3],STIM)

            SHAPE = samples.shape

            #generate random numbers for interpolation
            #nsam,nin,nhid
            EE = np.random.rand(SHAPE[0],1,1)
            gradpoints = EE*data + (1. - EE)*samples

            #compute the update
            dloss = D_step(data,samples,gradpoints)

            F = open(LOSS,"a")
            F.write("{},{}\n".format(gloss,dloss))
            F.close()


        SS = F_gen()
        gloss = G_step(SS[0],SS[1],SS[2],SS[3],STIM)

        F = open(LOSS,"a")
        F.write("{},{}\n".format(gloss,dloss))
        F.close()
        
        if k%1==0:
            ndm = 10
            dl = np.round(RF_low.get_value(),ndm)
            dd = np.round(RF_del.get_value(),ndm)
            dj = np.round(Js.get_value(),ndm)
            dt = np.round(THR.get_value(),ndm)
            dtd = np.round(THR_del.get_value(),ndm)
            aas = np.round(As.get_value(),ndm)
            OUT = "{}\t{}\t{}\t{}\t{}\t{}".format(dl,dd,dj,dt,dtd,aas)

            print(OUT)

            F = open(LOG,"a")
            F.write(OUT + "\n")
            F.close()

    
def train_gan(D_step,G_step,G_in,F_gen,STIM,INSHAPE,tag,use_data):

    print("Training RGAN")
        
    LOG = "./FF_logs/FF_log_"+tag+".csv"
    LOSS = "./FF_logs/FF_losslog_"+tag+".csv"

    F = open(LOG,"w")
    F.write("\tdRF\tdRFd\tdJ\tth\tdth\n")
    F.close()


    F = open(LOSS,"w")
    F.write("gloss,dloss\n")
    F.close()

    print("{}\t{}\t{}\t{}\t{}\n".format(RF_low.get_value(),
                                       RF_del.get_value(),
                                       Js.get_value(),
                                       THR.get_value(),
                                       THR_del.get_value()))

    print("dRF\tdRFd\tdJ\tdth")

    gloss = 10.
    dloss = 10.

    print("DATA {}".format(use_data))

    for k in range(niter):
    
        SS = F_gen()
        DD = F_gen()
        
        #get the disc inputs
        data = np.reshape(curves[np.random.choice(np.arange(len(curves)),INSHAPE[0])],INSHAPE)

        samples = G_in(SS[0],SS[1],SS[2],SS[3],STIM)

        #compute the update
        dloss = D_step(data,samples)
        gloss = G_step(SS[0],SS[1],SS[2],SS[3],STIM)

        F = open(LOSS,"a")
        F.write("{},{}\n".format(gloss,dloss))
        F.close()
        
        if k%1==0:
            ndm = 10
            dl = np.round(RF_low.get_value(),ndm)
            dd = np.round(RF_del.get_value(),ndm)
            dj = np.round(Js.get_value(),ndm)
            dt = np.round(THR.get_value(),ndm)
            dtd = np.round(THR_del.get_value(),ndm)
            OUT = "{}\t{}\t{}\t{}\t{}".format(dl,dd,dj,dt,dtd)

            print(OUT)

            F = open(LOG,"a")
            F.write(OUT + "\n")
            F.close()

def make_train_funcs(generator, Gparams, Ginputs, layers, INSHAPE,mode):

    if mode == "WGAN":
        return make_WGAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE)
    elif mode =="GAN":
        return make_GAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE)

def make_WGAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE):
    D_D_input = T.tensor3("DDinput","float32")
    D_G_input = T.tensor3("DGinput","float32")    

    discriminator = SD.make_net(INSHAPE, "WGAN", layers)

    #get the outputs
    D_dat_out = lasagne.layers.get_output(discriminator, T.log(1. + D_D_input))
    D_sam_out = lasagne.layers.get_output(discriminator, T.log(1. + D_G_input))
    G_sam_out = lasagne.layers.get_output(discriminator, T.log(1. + generator))

    #make the loss functions

    D_grad_input = T.tensor3("D_sam","float32")
    D_grad_out = lasagne.layers.get_output(discriminator, D_grad_input)

    Dgrad = T.sqrt((T.jacobian(T.reshape(D_grad_out,[-1]),D_grad_input)**2).sum(axis = [1,2,3]))
    Dgrad_penalty = ((Dgrad - 1.)**2).mean()

    lam = 1.

    Wdist = D_sam_out.mean() - D_dat_out.mean()
    D_loss_exp = D_sam_out.mean() - D_dat_out.mean() + lam*Dgrad_penalty#discriminator loss
    G_loss_exp = - G_sam_out.mean()#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .01
    
    b1 = .5
    b2 = .9

    D_updates = lasagne.updates.adam(D_loss_exp, lasagne.layers.get_all_params(discriminator, trainable=True), lr,beta1 = b1,beta2 = b2)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp, Gparams, lr,beta1 = b1,beta2 = b2)

    G_train_func = theano.function(Ginputs,G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')
    D_train_func = theano.function([D_D_input,D_G_input,D_grad_input],Wdist,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    Dout_func = theano.function([D_D_input,D_G_input],[D_sam_out,D_dat_out],allow_input_downcast = True,on_unused_input = 'ignore')

    return D_train_func, G_train_func,Dout_func

def make_GAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE):
    D_D_input = T.tensor3("DDinput","float32")
    D_G_input = T.tensor3("DGinput","float32")    

    discriminator = SD.make_net(INSHAPE, "WGAN", layers)

    #get the outputs
    D_dat_out = lasagne.layers.get_output(discriminator, T.log(1. + D_D_input))
    D_sam_out = lasagne.layers.get_output(discriminator, T.log(1. + D_G_input))
    G_sam_out = lasagne.layers.get_output(discriminator, T.log(1. + generator))

    #make the loss functions

    eps = .0001
    m1 = 1.-eps

    D_loss_exp = - T.log(eps + m1*D_dat_out.mean()) - T.log(1. -  m1*D_sam_out.mean())#discriminator loss
    G_loss_exp = - T.log(eps + m1*G_sam_out.mean())#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .001
    
    b1 = .5
    b2 = .9

    D_updates = lasagne.updates.adam(D_loss_exp, lasagne.layers.get_all_params(discriminator, trainable=True), lr,beta1 = b1,beta2 = b2)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp, Gparams, lr,beta1 = b1,beta2 = b2)

    G_train_func = theano.function(Ginputs,G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    D_train_func = theano.function([D_D_input,D_G_input],D_loss_exp,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    Dout_func = theano.function([D_D_input,D_G_input],[D_sam_out,D_dat_out],updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    return D_train_func, G_train_func, Dout_func

if __name__ == "__main__":

    run_GAN()
