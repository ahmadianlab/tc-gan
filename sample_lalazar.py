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
    n = 0
    for l in f:
        temp = l.split("\t")
        temp[-1] = temp[-1][:-1]

        if n != 1:
            out.append([float(t) for t in temp])
        else:
            out.append([t for t in temp])
        n += 1

    f.close()
        
    return np.array(out)

np.random.seed(1)

FN = sys.argv[1]

data = read_dat(sys.argv[1])

box_width = 40
print(box_width)

RF_low = theano.shared(np.float32(1),name = "RF_s")
RF_del = theano.shared(np.float32(1),name = "mean_W")
THR = theano.shared(np.float32(10),name = "s_W")
THR_del = theano.shared(np.float32(np.log(5.)),name = "s_W")
Js = theano.shared(np.float32(np.log(5000.)),name = "mean_b")

def set_param(PP):
    print(PP)

    RF_low.set_value(float(PP[0]))
    RF_del.set_value(float(PP[1]))
    Js.set_value(float(PP[2]))
    THR.set_value(float(PP[3]))
    THR_del.set_value(float(PP[4]))

def sample(par,o_gen,n_gen,fname,n = 20):
    out = []

    set_param(par)

    for k in range(n):
        print(k)
        sam = n_gen()
    
        TC2 = o_gen(sam)

        out.append(TC2)

    out = np.concatenate(out)

    np.savetxt("./FF_tuning_curve_samples/" + fname + ".csv",out)

def sample_log(name,log,o_gen,n_gen,n = 20):
    
    pars = [log[2 + (2**k)] for k in range(14)]

    for p in range(len(pars)):
        sample(pars[p],o_gen,n_gen,name + "_" + str(2**p),n)
    

def run_GAN():

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

    PARAM = [RF_low,RF_del,THR,THR_del,Js]

    FFout_sam = IO_func.get_FF_output(T.exp(RF_low),T.exp(RF_del),THR,T.exp(THR_del),T.exp(Js),receptive_widths,feedforward_conn,feedforward_strn,feedforward_thrs,pos,stimulus,nsam,nx,ny,nz,nhid,ni,dx)

    GINP = [feedforward_conn,feedforward_strn,receptive_widths,feedforward_thrs,stimulus]

    output_sam = theano.function(GINP,FFout_sam,allow_input_downcast = True,on_unused_input = 'ignore')
   
    name = FN.split("/")[-1].split(".")[0]

    print(name)
 
    sample_log(name,data,lambda x:np.reshape(np.transpose(output_sam(x[0],x[1],x[2],x[3],STIM),[0,2,1]),[-1,NI]),generate_samples,n = 30)

if __name__ == "__main__":

    run_GAN()
