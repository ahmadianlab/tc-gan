import theano
import numpy as np

if 0:
    F = open("./tuning_curveswgan_FF_LN_fit_40_noJ_40.csv","r")

    k = []
    mm = 0
    for l in F:
        
        if mm == 0:
            mm+=1
            continue 
        
        
        temp = l.split(",")
        if len(temp) != 27:
            k.append("bad")
            continue
        temp[-1] = temp[-1][:-1]

#    print temp
        
        for t in range(len(temp)):
            if temp[t][:2] == "--":
                temp[t] = temp[t][1:]
                

        k.append([float(x) for x in temp])

        if np.any(np.isnan(k[-1])):
            
            for m in k[-25:]:
                print(m)
            exit()
L = []
MAX = []
m = 0

for k in range(500):
    L.append(np.load("./disc_params/D_par_{}_wgan_FF_LN_fit_40_noJ_40.npy".format(k)))
    d = 0
    for j in L[k]:
        d = np.max([d,np.max(np.abs(j))])
        m = np.max([m,np.max(np.abs(j))])
    MAX.append(d)
    print(d)

    if np.isnan(d):
        break

print(m)

np.savetxt("./discpar.csv",L)
