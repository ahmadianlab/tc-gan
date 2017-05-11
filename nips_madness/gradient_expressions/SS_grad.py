import theano
import theano.tensor as T
import lasagne as L
import numpy as np

from ..weight_gen import weight, generate_weight
from ..ssnode import rate_to_volt

#what I need is: to write an expression that takes a variable R and matrix W (ss response and weight matrix) and returns the gradient.
#Now I need functions that take the parameters and compute Wgrads w.r.t. the parameters

def rectify(x):
    return .5*(x + abs(x))

def WRgrad_batch(R,W,DW,I,n,k,nz,nb,N,io_type = "asym_tanh",r0 = 100.,r1 = 200.):
    '''

    Implementation with no cutoff. Powerlaw nonlinearity extends to infinity

    This expression takes theano variables for R,W,dW/dth,I,n, and k and outputs the gradient dr/dth
    
    This one is set up to handle batches of [nz,nb]

    R - [nz,nb,2N]
    W - [nz,2N,2N]
    DW - [nz,2N,2N,2,2]
    I - [nb,2N]
    n - scalar
    k - scalar


    dr/dw = 
    '''    

    wt = T.reshape(W,[nz,1,2*N,2*N])
    rt = T.reshape(R,[nz,nb,1,2*N])

    V = (wt*rt).sum(axis = 3) + T.reshape(I,[1,nb,2*N]) #[nz,nb,2N]
    V_clipped = rectify(V)

    if io_type == "asym_power":
        phi = power_phi(V,k,n,nz,nb,N)
    elif io_type == "asym_linear":
        phi = linear_phi(V,k,n,r0,nz,nb,N)
    elif io_type == "asym_tanh":
        phi = tanh_phi(V,k,n,r0,r1,nz,nb,N)
    else:
        print("Must specify a valid io_type! Not {}.".format(io_type))
        exit()
    
    Wall = T.reshape(W,[-1,2*N,2*N])
    WIt,up = theano.map(lambda x:T.identity_like(x),[Wall])
    WIt = T.reshape(WIt,[nz,1,2*N,2*N])#identity matrices for W

    J = WIt - phi*wt# [nz,nb,2*N,2*N]

    dwt = T.reshape(DW,[nz,1,2*N,2*N,2,2])

    rt = T.reshape(R,[nz,nb,1,2*N,1,1])

    B = T.reshape(T.reshape(phi,[nz,nb,2*N,1,1])*((dwt*rt).sum(axis = 3)),[nz,nb,1,2*N,2,2]) #DW [nz,nb,2N,2N,2,2]

    MI,up = theano.map(lambda x:T.nlinalg.MatrixInverse()(x),[T.reshape(J,[-1,2*N,2*N])])
    MI = T.reshape(MI,[nz,nb,2*N,2*N,1,1])

    return (MI*B).sum(axis = 3)

def tanh_phi(V,k,n,r0,r1,nz,nb,N):
    v0 = rate_to_volt(r0, k, n)
    
    V_clipped = rectify(V)

    V_low = V_clipped*(V_clipped <= v0)
    V_high = V_clipped*(V_clipped > v0)

    VH_arg = (n*r0/v0)*(V_high - v0)/(r1 - r0)

    phi_low = T.reshape(n*k*T.power(V_low,n - 1.),[nz,nb,2*N,1]) #[nz,nb,2N]
    phi_high = T.reshape((n*r0/v0)/T.power(T.cosh(VH_arg),2.),[nz,nb,2*N,1]) #[nz,nb,2N]

    return phi_high + phi_low

def linear_phi(V,k,n,r0,nz,nb,N):
    V_clipped = T.clip(V, 0., rate_to_volt(r0, k, n))
    return T.reshape(n*k*T.power(V_clipped,n - 1.),[nz,nb,2*N,1]) #[nz,nb,2N]

def power_phi(V,k,n,nz,nb,N):
    V_clipped = rectify(V)
    return T.reshape(n*k*T.power(V_clipped,n - 1.),[nz,nb,2*N,1]) #[nz,nb,2N]
