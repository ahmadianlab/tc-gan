import argparse
import json
from tc_gan.deafult_params import DEFAULT_PARAMS
parser = argparse.ArgumentParser()

parser.add_argument("GAN_data_path",type = str,help = "Path to the GAN fit data files.")
parser.add_argument("MM_data_path",type = str,help = "Path to the MM fit data files.")
                            
args = vars(parser.parse_args())

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

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

GAN_data_loc = args["GAN_data_path"]
MM_data_loc = args["MM_data_path"]

truth = json.load(open(GAN_data_loc + "info.json","r"))
Glearning = np.loadtxt(GAN_data_loc + "generator.csv",delimiter = ",",skiprows=1)
Mlearning = read_csv(MM_data_loc + "generator.csv",header = True)

GP = np.reshape(Glearning[:,1:],[-1,3,2,2])
MP = np.reshape(Mlearning[:,1:],[-1,3,2,2])

true_params = np.array([truth["run_config"]["true_ssn_options"]["D"],truth["run_config"]["true_ssn_options"]["J"],DEFAULT_PARAMS["S"]])

def sMAPE(x,y):
    return np.mean(100*2*np.abs(x - y)/(x+y),axis = (-1,-2,-3))

fig,ax = plt.subplots(2,3,figsize = (6,4))

ax[0,0].plot(sMAPE(GP,np.expand_dims(true_params,0)))
ax[1,0].plot(sMAPE(MP,np.expand_dims(true_params,0)))

ax[0,0].set_xlabel("Generator Gradient Step")
ax[0,0].set_ylabel("GAN sMAPE")
ax[1,0].set_xlabel("Generator Gradient Step")
ax[1,0].set_ylabel("MM sMAPE")
col = ["r","g","b","c"]
pars = [GP,MP]
epc = [Glearning[:,0],Mlearning[:,0]]
dl = [0,.05,.1,.15]
for i in range(2):
    for j in range(2):
        for k in range(2):
            fe = epc[k][-1]
            
            ii = i + 2*j
            cc = col[ii]
            ax[k,1].plot(epc[k],pars[k][:,0,i,j],color = cc)
            ax[k,1].fill_between(epc[k],pars[k][:,0,i,j] - pars[k][:,1,i,j],pars[k][:,0,i,j] + pars[k][:,1,i,j],alpha = .2,color = cc)
            ax[k,1].set_xlim(0,epc[k][-1]*1.25)
            
            ax[k,1].plot([fe*(1. + dl[ii]),fe*(1. + dl[ii] + .1)],[true_params[0,i,j],true_params[0,i,j]],color = cc)
            ax[k,1].plot([fe*(1.05 + dl[ii]),fe*(1.05 + dl[ii])],[true_params[0,i,j]-true_params[1,i,j],true_params[0,i,j]+true_params[1,i,j]],color = cc)

ax[0,1].set_xlabel("Generator Gradient Step")
ax[0,1].set_ylabel("J \pm \delta J")

ax[1,1].set_xlabel("Generator Gradient Step")
ax[1,1].set_ylabel("J \pm \delta J")

for i in range(2):
    for j in range(2):
        for k in range(2):
            fe = epc[k][-1]
            
            ii = i + 2*j
            cc = col[ii]
            ax[k,2].plot(epc[k],pars[k][:,2,i,j],color = cc)
            ax[k,2].set_xlim(0,epc[k][-1]*1.25)
            
            ax[k,2].plot([fe*(1.05),fe*(1.05+ .1)],[true_params[2,i,j],true_params[2,i,j]],color = cc)
ax[0,2].set_xlabel("Generator Gradient Step")
ax[0,2].set_ylabel("\sigma")

ax[1,2].set_xlabel("Generator Gradient Step")
ax[1,2].set_ylabel("\sigma")

plt.tight_layout()   
plt.savefig("./Fig5pythonversion.pdf")