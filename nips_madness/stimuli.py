import numpy as np

def sigm(x,l = .1):
    return 1./(1 + np.exp(-x/l))

def band(x,b,l = .1):
    return sigm(x + (b/2),l)*sigm((b/2) - x,l)

def input(bv,x,l = .1,c = [20.],o = [0.]):
    return np.concatenate([con*np.array([np.concatenate([band(x-off,b,l),band(x-off,b,l)]) for b in bv]) for con in c for off in o])
