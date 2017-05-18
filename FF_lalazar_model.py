import lasagne 
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD
import math 
import sys 
import time
import FF_functions.lalazar_func as IO_func

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

MODE = sys.argv[1]
DATA = bool(sys.argv[2])
box_width = int(sys.argv[3])
RF_i = float(sys.argv[4])

if MODE not in ["WGAN","GAN"]:
    print("Mode not recognized")
    exit()

tag = MODE + "_" + str(DATA) + "_" + str(int(RF_i*10)) + "_new_"

print(tag)

#if len(sys.argv) == 5:
#    LOG = read_log(sys.argv[4])
#    start_params = LOG[-1]
#else:

start_params = [np.log(RF_i),np.log(.25),np.log(5500),np.log(10.),np.log(10.)]

#import the data
curves = read_dat("lalazar_data/TuningCurvesFull_Pronation.dat")
curves = np.array([c/(np.mean(c) + .0001) for c in curves])

X_pos = read_dat("lalazar_data/XCellsFull.dat")
Y_pos = read_dat("lalazar_data/YCellsFull.dat")
Z_pos = read_dat("lalazar_data/ZCellsFull.dat")
#done importing

RF_low = theano.shared(np.float32(start_params[0]),name = "RF_s")
RF_del = theano.shared(np.float32(start_params[1]),name = "mean_W")
THR = theano.shared(np.float32(start_params[3]),name = "s_W")
THR_del = theano.shared(np.float32(start_params[4]),name = "s_W")
Js = theano.shared(np.float32(start_params[2]),name = "mean_b")

T_RF_low = theano.shared(np.float32(np.log(.5)),name = "RF_s")
T_RF_del = theano.shared(np.float32(np.log(.1)),name = "mean_W")
T_THR = theano.shared(np.float32(0.),name = "s_W")
T_THR_del = theano.shared(np.float32(np.log(.1)),name = "s_W")
T_Js = theano.shared(np.float32(np.log(10.)),name = "mean_b")

#The full model takes RF widths from a uniform dist
#The full model takes the FF weights to be unifrom in [0,1] after sparsifying
#The full model applies the same threshold to all hidden units

#parameters:
# - RF_low
# - delta_RF
# - threshold
# - weight_scale (why not?)

#random saples:
# - RF width (uniform [0,1]) - [nsam,nx,ny,nz]
# - FF connections (discreete {0,1}) - [nsam,nhid,nx*ny*nz]
# - FF strengths (uniform [0,1]) - [nsam,nhid,nx*ny*nz]

def run_GAN(mode = MODE):

    dx = 1./box_width

    nsam = theano.shared(10)
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
        TH_s = np.random.rand(NSAM,NHID)

        return FF_c,FF_s,wid,TH_s

    stimulus = T.matrix("input","float32")

    feedforward_conn = T.tensor3("connts","float32")
    feedforward_strn = T.tensor3("streng","float32")
    feedforward_thrs = T.matrix("streng","float32")
    receptive_widths = T.tensor4("widths","float32")

    feedforward_conn_true = T.tensor3("connts_T","float32")
    feedforward_strn_true = T.tensor3("streng_T","float32")
    feedforward_thrs_true = T.matrix("streng","float32")
    receptive_widths_true = T.tensor4("widths_T","float32")


    PARAM = [RF_low,RF_del,THR,THR_del,Js]

    FFout_sam = IO_func.get_FF_output(T.exp(RF_low),T.exp(RF_del),THR,T.exp(THR_del),T.exp(Js),receptive_widths,feedforward_conn,feedforward_strn,feedforward_thrs,pos,stimulus,nsam,nx,ny,nz,nhid,ni,dx)

    GINP = [feedforward_conn,feedforward_strn,receptive_widths,feedforward_thrs,stimulus]

    output_sam = theano.function(GINP,FFout_sam,allow_input_downcast = True,on_unused_input = 'ignore')

    FFout_dat = IO_func.get_FF_output(T.exp(T_RF_low),T.exp(T_RF_del),T_THR,T.exp(T_THR_del),T.exp(T_Js),receptive_widths_true,feedforward_conn_true,feedforward_strn_true,feedforward_thrs_true,pos,stimulus,nsam,nx,ny,nz,nhid,ni,dx)
    
    DINP = [feedforward_conn_true,feedforward_strn_true,receptive_widths_true,feedforward_thrs_true,stimulus]

    output_dat = theano.function(DINP,FFout_dat,allow_input_downcast = True,on_unused_input = 'ignore')


    ##print out some sample tuning curves
    TT = time.time()

    out = []

    for k in range(10):
        print(k)
        sam = generate_samples()
    
        TC2 = output_sam(sam[0],sam[1],sam[2],sam[3],STIM)

        TC2 = np.reshape(np.transpose(TC2,[0,2,1]),[-1,NI])

        out.append(TC2)

    out = np.concatenate(out)
    np.savetxt("./FF_tuning_curve_samples/FF_ll_tuning_curves_sam.csv",out)

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

    FOBS_dat = (T.reshape(mask,[1,1,NOBS,nhid])*T.reshape(FFout_dat,[nsam,ni,1,nhid])).sum(axis = 3)
    FOBS_sam = (T.reshape(mask,[1,1,NOBS,nhid])*T.reshape(FFout_sam,[nsam,ni,1,nhid])).sum(axis = 3)

    get_DD_input = theano.function(DINP,FOBS_dat,allow_input_downcast = True)
    get_DG_input = theano.function(GINP,FOBS_sam,allow_input_downcast = True)

    #I need a function that returns training functions
    D_train, G_train, DOUT = make_train_funcs(FOBS_sam,PARAM,GINP,[128,128],(NSAM,NI,NOBS),mode)

    print tag

    #now define the actual values and test
    train(D_train,G_train,get_DD_input,get_DG_input,generate_samples,STIM,(NSAM,NI,NOBS),mode,tag + str(box_width),DATA)

def train(D_step,G_step,D_in,G_in,F_gen,STIM,INSHAPE,mode,tag,data):
    
    if mode =="WGAN":
        train_wgan(D_step,G_step,D_in,G_in,F_gen,STIM,INSHAPE,tag,data)
    elif mode =="GAN":
        train_gan(D_step,G_step,D_in,G_in,F_gen,STIM,INSHAPE,tag,data)

def train_wgan(D_step,G_step,D_in,G_in,F_gen,STIM,INSHAPE,tag,use_data):
   
    print("Trainign WGAN")
 
    NDstep = 5

    LOG = "./FF_logs/FF_log_"+tag+".csv"
    LOSS = "./FF_logs/FF_losslog_"+tag+".csv"
 
    F = open(LOG,"w")
    F.write("{}\t{}\t{}\t{}\t{}\n".format(
            T_RF_low.get_value(),
            T_RF_del.get_value(),
            T_Js.get_value(),
            T_THR.get_value(),
            T_THR_del.get_value()))
    F.write("RF\tRFd\tJ\tth\tth_d\n")
    F.close()

    F = open(LOSS,"w")
    F.write("gloss,dloss\n")
    F.close()


    print("{}\t{}\t{}\t{}\t{}\n".format(T_RF_low.get_value(),
                                       T_RF_del.get_value(),
                                       T_Js.get_value(),
                                       T_THR.get_value(),
                                       T_THR_del.get_value()))
   
    print("{}\t{}\t{}\t{}\t{}\n".format(RF_low.get_value(),
                                       RF_del.get_value(),
                                       Js.get_value(),
                                       THR.get_value(),
                                       THR_del.get_value()))
   
    print("dRF\tdRFd\tdJ\tdth\tth_d")

    gloss = 10.
    dloss = 10.

    for k in range(niter):
    
        for dstep in range(NDstep):
            #get the samples for training
            SS = F_gen()
            DD = F_gen()

            #get the data inputs

            if use_data:
                data = np.reshape(curves[np.random.choice(np.arange(len(curves)),INSHAPE[0])],INSHAPE)
            else:
                data = D_in(DD[0],DD[1],DD[2],DD[3],STIM)

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
            OUT = "{}\t{}\t{}\t{}\t{}".format(dl,dd,dj,dt,dtd)

            print(OUT)

            F = open(LOG,"a")
            F.write(OUT + "\n")
            F.close()

    
def train_gan(D_step,G_step,D_in,G_in,F_gen,STIM,INSHAPE,tag,use_data):

    print("Training RGAN")
        
    LOG = "./FF_logs/FF_log_"+tag+".csv"
    LOSS = "./FF_logs/FF_losslog_"+tag+".csv"

    F = open(LOG,"w")
    F.write("{}\t{}\t{}\t{}\t{}\n".format(T_RF_low.get_value(),
                                       T_RF_del.get_value(),
                                       T_Js.get_value(),
                                       T_THR.get_value(),
                                       T_THR_del.get_value()))
    F.write("\tdRF\tdRFd\tdJ\tth\tdth\n")
    F.close()


    F = open(LOSS,"w")
    F.write("gloss,dloss\n")
    F.close()

    print("{}\t{}\t{}\t{}\t{}\n".format(T_RF_low.get_value(),
                                       T_RF_del.get_value(),
                                       T_Js.get_value(),
                                       T_THR.get_value(),
                                       T_THR_del.get_value()))

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
        if use_data:
            data = np.reshape(curves[np.random.choice(np.arange(len(curves)),INSHAPE[0])],INSHAPE)
        else:
            data = D_in(DD[0],DD[1],DD[2],DD[3],STIM)

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
    
    D_dat = SD.make_net(np.log(1. + D_D_input),INSHAPE,"WGAN",layers)
    D_sam = SD.make_net(np.log(1. + D_G_input),INSHAPE,"WGAN",layers,params = lasagne.layers.get_all_layers(D_dat))
    G_sam = SD.make_net(np.log(1. + generator),INSHAPE,"WGAN",layers,params = lasagne.layers.get_all_layers(D_dat))
    
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

    lam = 100.

    Wdist = D_sam_out.mean() - D_dat_out.mean()
    D_loss_exp = D_sam_out.mean() - D_dat_out.mean() + lam*Dgrad_penalty#discriminator loss
    G_loss_exp = - G_sam_out.mean()#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .001
    
    b1 = .5
    b2 = .9

    D_updates = lasagne.updates.adam(D_loss_exp, lasagne.layers.get_all_params(D_dat), lr,beta1 = b1,beta2 = b2)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp, Gparams, lr,beta1 = b1,beta2 = b2)

    G_train_func = theano.function(Ginputs,G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')
    D_train_func = theano.function([D_D_input,D_G_input,D_grad_input],Wdist,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    Dout_func = theano.function([D_D_input,D_G_input],[D_sam_out,D_dat_out],allow_input_downcast = True,on_unused_input = 'ignore')

    return D_train_func, G_train_func,Dout_func

def make_GAN_funcs(generator, Gparams, Ginputs, layers, INSHAPE):
    D_D_input = T.tensor3("DDinput","float32")
    D_G_input = T.tensor3("DGinput","float32")    
    
    D_dat = SD.make_net(T.log(1. + D_D_input),INSHAPE,"CE",layers)
    D_sam = SD.make_net(T.log(1. + D_G_input),INSHAPE,"CE",layers,params = lasagne.layers.get_all_layers(D_dat))
    G_sam = SD.make_net(T.log(1. + generator),INSHAPE,"CE",layers,params = lasagne.layers.get_all_layers(D_dat))
    
    #get the outputs
    D_dat_out = lasagne.layers.get_output(D_dat)
    D_sam_out = lasagne.layers.get_output(D_sam)
    G_sam_out = lasagne.layers.get_output(G_sam)
    
    #make the loss functions

    eps = .0001
    m1 = 1.-eps

    D_loss_exp = - T.log(eps + m1*D_dat_out.mean()) - T.log(1. -  m1*D_sam_out.mean())#discriminator loss
    G_loss_exp = - T.log(eps + m1*G_sam_out.mean())#generative loss
    
    #we can just use lasagne/theano derivatives to get the grads for the discriminator
    lr = .001
    
    b1 = .5
    b2 = .9

    D_updates = lasagne.updates.adam(D_loss_exp, lasagne.layers.get_all_params(D_dat), lr,beta1 = b1,beta2 = b2)#discriminator training function
    G_updates = lasagne.updates.adam(G_loss_exp, Gparams, lr,beta1 = b1,beta2 = b2)

    G_train_func = theano.function(Ginputs,G_loss_exp,updates = G_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    D_train_func = theano.function([D_D_input,D_G_input],D_loss_exp,updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    Dout_func = theano.function([D_D_input,D_G_input],[D_sam_out,D_dat_out],updates = D_updates,allow_input_downcast = True,on_unused_input = 'ignore')

    return D_train_func, G_train_func, Dout_func

if __name__ == "__main__":

    run_GAN()
