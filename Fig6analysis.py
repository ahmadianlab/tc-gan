import argparse
import json
from tc_gan.deafult_params import DEFAULT_PARAMS
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def read_csv(f,header = False,skip = 1):
    F = open(f,"r")
    out = []
    n = -1
    for l in F:
        n += 1
        if n%skip == 0:
            temp = l.split(",")
            if header and n == 0:
                continue#out.append([t.split("\n")[0] for t in temp])
            else:
                out.append([float(t.split("\n")[0]) for t in temp])

        
    out = np.array(out)[:1000*int(len(out)/1000)]
    print(out.shape)
    return out

ganf = glob.glob("./scripts/fig6/gan/*")
mmf = glob.glob("./scripts/fig6/mm/*")

truth = json.load(open(ganf[0] + "info.json","r"))
Glearning = [np.loadtxt(g + "generator.csv",delimiter = ",",skiprows=1) for g in ganf]
Mlearning = [np.loadtxt(g + "generator.csv",delimiter = ",",skiprows=1) for f in mmf]

GP = [np.reshape(g[:,1:],[-1,3,2,2]) for g in Glearning]
MP = [np.reshape(m[:,1:],[-1,3,2,2]) for m in Mlearning]

true_params = np.array([truth["run_config"]["true_ssn_options"]["D"],truth["run_config"]["true_ssn_options"]["J"],DEFAULT_PARAMS["S"]])

def sMAPE(x,y):
    return np.mean(100*2*np.abs(x - y)/(x+y),axis = (-1,-2,-3))

fig,ax = plt.subplots(2,3,figsize = (6,4))

gans = [sMAPE(g,np.expand_dims(true_params,0))[-1] for g in GP]
mms = [sMAPE(g,np.expand_dims(true_params,0))[-1] for g in MP]

plt.hist(gans,np.linspace(0,100,30),normed = True,label = "GAN")
plt.hist(mms,np.linspace(0,100,30),normed = True,label = "moment matching")
plt.xlabel("Final sMAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savegif("./Fig6pythonversion.pdf")