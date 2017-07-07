import lasagne
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD
import math 
import sys 
import time
import FF_functions.lalazar_func as IO_func


# Use single precision (even when it is running in CPU):
theano.config.floatX = 'float32'
# [Note] As this script assumes float32 everywhere but Lasagne and
# simple_discriminator do not, this setting is required to make it
# work.  Since altering theano.config.<attribute> is not recommended
# by the theano manual, it may be a good idea to remove the hard coded
# assumption of the default data type.
def max_min_par(P):
    m = 0

    for p in P:
        m = np.max([m,np.max(np.reshape(np.abs(p),[-1]))])

    print m

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

box_width = int(sys.argv[1])

tag = "wgan_FF_" + str(box_width) + "_"

#tag += "slow_"

print(tag)

start_params = [np.log(2.),np.log(.3),np.log(1000),0.,np.log(1.),np.log(1.)]

#import the data
curves = read_dat("lalazar_data/TuningCurvesFull_Pronation.dat")
#curves = np.array([[x if x > 5 else 0 for x in tc] for tc in curves])
#curves = np.array([(x - x.mean())/(x.max() - x.min() + .001) for x in curves])

X_pos = read_dat("lalazar_data/XCellsFull.dat")
Y_pos = read_dat("lalazar_data/YCellsFull.dat")
Z_pos = read_dat("lalazar_data/ZCellsFull.dat")
#done importing

RF_low = theano.shared(np.float32(start_params[0]),name = "RF_s")
RF_del = theano.shared(np.float32(start_params[1]),name = "mean_W")
THR = theano.shared(np.float32(start_params[3]),name = "s_W")
THR_del = theano.shared(np.float32(start_params[4]),name = "s_W")
Js = theano.shared(np.float32(start_params[2]),name = "mean_b")
As = theano.shared(np.float32(start_params[5]),name = "mean_b")

PARAM = [RF_low,RF_del,THR,THR_del,Js]

def run_GAN(mode = "WGAN"):

    print("Start")

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
      
    pos = theano.shared(np.array([[[[x,y,z] for z in ZZ] for y in YY] for x in XX]).astype("float32"))

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
        wid = np.random.rand(NSAM,NX,NY,NZ)
        FF_c = np.zeros((NSAM,NHID,NX*NY*NZ)) 

        samind = np.arange(NX*NY*NZ)

        for s in np.arange(NSAM):
            for h in np.arange(NHID):
                FF_c[s,h,np.random.choice(samind,NFF)] = 1

        FF_s = np.random.rand(NSAM,NHID,NX*NY*NZ)
        TH_s = np.random.uniform(-1,1,[NSAM,NHID])

        return FF_c,FF_s,wid,TH_s

    stimulus = T.matrix("input","float32")

    feedforward_conn = T.tensor3("connts","float32")
    feedforward_strn = T.tensor3("streng","float32")
    feedforward_thrs = T.matrix("streng","float32")
    receptive_widths = T.tensor4("widths","float32")

    FFout_sam = IO_func.get_FF_output(T.exp(RF_low),T.exp(RF_del),THR,T.exp(THR_del),T.exp(Js),T.exp(As),receptive_widths,feedforward_conn,feedforward_strn,feedforward_thrs,pos,stimulus,nsam,nx,ny,nz,nhid,ni,dx)

    GINP = [feedforward_conn,feedforward_strn,receptive_widths,feedforward_thrs,stimulus]

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

    print("training funcs")

    #I need a function that returns training functions
    D_train, G_train, DOUT, Dparams = make_train_funcs(FOBS_sam,PARAM,GINP,[128,128],(NSAM,NI,NOBS),mode)

    print(tag)

    #now define the actual values and test
    train(D_train,G_train,get_DG_input,generate_samples,Dparams,STIM,(NSAM,NI,NOBS),mode,tag + str(box_width))

def train(D_step,G_step,G_in,F_gen,Dparams,STIM,INSHAPE,mode,tag,NDstep = 5):
    
    if mode =="WGAN":
        train_wgan(D_step,G_step,G_in,F_gen,Dparams,STIM,INSHAPE,tag,NDstep)
    elif mode =="GAN":
        train_gan(D_step,G_step,G_in,F_gen,Dparams,STIM,INSHAPE,tag)

def train_wgan(D_step,G_step,G_in,F_gen,Dparams,STIM,INSHAPE,tag,NDstep = 5):
   
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

    tc_file = "./tuning_curves" + tag + ".csv"

    F = open(tc_file,"w")
    F.write("tuning curves for " + tag)
    F.close()


    for k in range(niter):

        np.save("disc_params/D_par_{}_".format(k) + tag,lasagne.layers.get_all_param_values(Dparams))
    
        for dstep in range(1000 if k == 0 else NDstep):
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

            max_min_par(lasagne.layers.get_all_param_values(Dparams))


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

            curv = np.reshape(G_in(SS[0],SS[1],SS[2],SS[3],STIM),[20,27])

            F = open(tc_file,"a")

            for c in curv:
                l = str(c[0])
                for val in c[1:]:
                    l += "," + str(val)

                l += "\n"
                F.write(l)

            F.close()

    
def train_gan(D_step,G_step,G_in,F_gen,STIM,INSHAPE,tag,):

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


    for k in range(niter):

        np.save("disc_params/D_par_{}_".format(k) + tag,lasagne.layers.get_all_param_values(Dparams))
    
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

    discriminator = SD.make_net(INSHAPE, "WGAN")

    Dparameters = discriminator

    #get the outputs
#    D_dat_out = lasagne.layers.get_output(discriminator, T.log(1. + D_D_input))
#    D_sam_out = lasagne.layers.get_output(discriminator, T.log(1. + D_G_input))
#    G_sam_out = lasagne.layers.get_output(discriminator, T.log(1. + generator))

    D_dat_out = lasagne.layers.get_output(discriminator, D_D_input)
    D_sam_out = lasagne.layers.get_output(discriminator, D_G_input)
    G_sam_out = lasagne.layers.get_output(discriminator, generator)
    
    #make the loss functions
    
    D_grad_input = T.tensor3("D_sam","float32")

    D_grad_out = lasagne.layers.get_output(discriminator, T.log(1 + D_grad_input))

    Dgrad = T.sqrt((T.jacobian(T.reshape(D_grad_out,[-1]),D_grad_input)**2).sum(axis = [1,2,3]))
    Dgrad_penalty = ((Dgrad - 1.)**2).mean()

    dpars = lasagne.layers.get_all_params(discriminator)

    plam = .00001
    Ploss = (dpars[0]**2).sum()

    for p in dpars[1:]:
        Ploss += (p**2).sum()

    lam = 10.

    Wdist = D_sam_out.mean() - D_dat_out.mean()
    D_loss_exp = D_sam_out.mean() - D_dat_out.mean() + lam*Dgrad_penalty + plam * Ploss#discriminator loss

    G_loss_exp = - G_sam_out.mean()#generative loss

    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .001
    
    b1 = .5
    b2 = .9

    D_updates = lasagne.updates.adam(D_loss_exp, lasagne.layers.get_all_params(discriminator, trainable=True), lr,beta1 = b1,beta2 = b2)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp, Gparams, lr,beta1 = b1,beta2 = b2)

    G_train_func = theano.function(Ginputs,G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')
    D_train_func = theano.function([D_D_input,D_G_input,D_grad_input],Wdist,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    Dout_func = theano.function([D_D_input,D_G_input],[D_sam_out,D_dat_out],allow_input_downcast = True,on_unused_input = 'ignore')

    return D_train_func, G_train_func,Dout_func, Dparameters

def make_GAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE):
    D_D_input = T.tensor3("DDinput","float32")
    D_G_input = T.tensor3("DGinput","float32")    

    discriminator = SD.make_net(INSHAPE, "CE", layers)

    Dparameters = discriminator

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

    return D_train_func, G_train_func, Dout_func, Dparameters

if __name__ == "__main__":

    run_GAN()
