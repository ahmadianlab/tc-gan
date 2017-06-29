import lasagne 
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD
import math 
import sys 
import time
import FF_functions.lalazar_func as IO_func

theano.config.floatX = "float64"

box_width = 40

RF_low = theano.shared(np.float32(1),name = "RF_s")
RF_del = theano.shared(np.float32(1),name = "mean_W")
THR = theano.shared(np.float32(10),name = "s_W")
THR_del = theano.shared(np.float32(np.log(5.)),name = "s_W")
Js = theano.shared(np.float32(np.log(5000.)),name = "mean_b")
As = theano.shared(np.float32(np.log(0)),name = "dist")

params = [RF_low,RF_del,THR,THR_del,Js,As]

def set_param(PP):

    for p in range(len(PP)):
        if p != 2:
            params[p].set_value(np.float32(np.log(PP[p])))    
        else:
            params[p].set_value(np.float32(PP[p]))    

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
        
        TH_s = np.random.uniform(-1,1,[NSAM,NHID])

        return FF_c,FF_s,wid,TH_s

    stimulus = T.matrix("input","float32")

    feedforward_conn = T.tensor3("connts","float32")
    feedforward_strn = T.tensor3("streng","float32")
    feedforward_thrs = T.matrix("streng","float32")
    receptive_widths = T.tensor4("widths","float32")

    PARAM = [RF_low,RF_del,THR,THR_del,Js,As]

    FFout_sam = IO_func.get_FF_output(T.exp(RF_low),T.exp(RF_del),THR,T.exp(THR_del),T.exp(Js),T.exp(As),receptive_widths,feedforward_conn,feedforward_strn,feedforward_thrs,pos,stimulus,nsam,nx,ny,nz,nhid,ni,dx)

    GINP = [feedforward_conn,feedforward_strn,receptive_widths,feedforward_thrs,stimulus]

    output_sam = theano.function(GINP,FFout_sam,allow_input_downcast = True,on_unused_input = 'ignore')
   
    rflR = [.01,3]
    rfdR = [.01,3]
    thrR = [-10,10]
    thdR = [1.,50]
    jR = [500,5000]

    def osam(x):
        return np.reshape(np.transpose(output_sam(x[0],x[1],x[2],x[3],STIM),[0,2,1]),[-1,NI])

    fpar = "./grid_search_params.csv"
    F = open(fpar,"w")
    F.write("rfl,rfd,ths,thd,j\n")
    F.close()
    
    fcur = "./grid_search_curves.csv"
    F = open(fcur,"w")
    F.write("rfl,rfd,ths,thd,j\n")
    F.close()

    def savedata(F,dat):
        F = open(F,"a")
        F.write("{}".format(dat[0]))
        for d in dat[1:]:
            F.write(",{}".format(d))
        F.write("\n")
        F.close()

    npar = 10
    nn = 0

    for rfl in np.linspace(rflR[0],rflR[1],20):
        for rfd in np.linspace(rfdR[0],rfdR[1],20):
            for jj in np.linspace(jR[0],jR[1],20):
                print(nn)
                t1 = time.time()
                par = [rfl,rfd,0,10,j]
                savedata(fpar,par)
                for k in range(10):
                    set_param(par)
                    sam = generate_samples()
                    
                    TC2 = osam(sam)
                    
                    for tc in TC2:
                        savedata(fcur,tc)
                        
                nn += 1
                print(nn,time.time() - t1)
                    
if __name__ == "__main__":

    run_GAN()
