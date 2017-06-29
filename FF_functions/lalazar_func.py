import lasagne 
import theano
import theano.tensor as T
import numpy as np
import discriminators.simple_discriminator as SD
import math 
import sys 
import time

theano.config.floatX = "float32"

def rectify(x):
    return np.float32(.5)*(x + abs(x))

def get_FF_output(RF_l,RF_d,TH,TH_d,J,a,RF_w,FF_con,FF_str,TH_sam,x,inp,nsam,nx,ny,nz,nhid,ni,dx):

    """

    RF_l - low end of the RF width distribution
    RF_d - width of RF distribuion
    TH   - threshold

    """

    pos = T.reshape(x,[1,-1,3])    
    stim = T.reshape(inp,[ni,1,3])
    dist_sq = T.reshape(((pos - stim)**2).sum(axis = 2),[1,ni,nx*ny*nz]) 
    widths = T.reshape(RF_w,[nsam,1,nx*ny*nz])
 
    exponent = dist_sq/(2*(widths*RF_d + RF_l)**2)#[nsam,ni,nx*y*nz]
#    input_activations = (dx**3)*T.reshape(T.exp(-exponent)/T.sqrt(((2*math.pi)**3)*(widths*RF_d + RF_l)**6),[nsam,ni,1,nx*ny*nz])
    input_activations = T.reshape(T.exp(-exponent),[nsam,ni,1,nx*ny*nz])
    input_activations = input_activations/T.sum(input_activations,axis = 3,keepdims = True)

    weights = T.reshape(J*FF_con*FF_str,[nsam,1,nhid,nx*ny*nz])
    
    hidden_activations = rectify((input_activations*weights).sum(axis = 3) - T.reshape((TH + (T.sgn(TH_sam)*(T.abs_(TH_sam)**a))*TH_d),[nsam,1,nhid]))#[nsam,ni,nhid]

    mean = hidden_activations.mean(axis = 1,keepdims = True)
    MAX = hidden_activations.max(axis = 1,keepdims = True)
    MIN = hidden_activations.min(axis = 1,keepdims = True)
    
    return (hidden_activations - mean)/(MAX - MIN + .001)

def get_FF_output_pruned(RF_l,RF_d,TH,TH_d,J,a,RF_w,FF_str,TH_sam,x,inp,nsam,nx,ny,nz,nhid,ni,dx):

    """

    RF_l - low end of the RF width distribution
    RF_d - width of RF distribuion
    TH   - threshold

    """

    W = T.reshape(RF_w,[nsam,1,-1])
    TH_s = T.reshape(TH_sam,[nsam,1,-1])

    pos = T.reshape(x,[nsam,1,-1,3])    
    stim = T.reshape(inp,[1,ni,1,3])
    dist_sq = T.reshape(((pos - stim)**2).sum(axis = 3),[nsam,ni,-1])
 
    exponent = dist_sq/(2*(W*RF_d + RF_l)**2)#[nsam,ni,nff]
#    input_activations = (dx**3)*T.reshape(T.exp(-exponent)/T.sqrt(((2*math.pi)**3)*(widths*RF_d + RF_l)**6),[nsam,ni,1,nx*ny*nz])
    input_activations = T.reshape(T.exp(-exponent),[nsam,ni,1,-1])
    inpurt_activations = input_activations/T.sum(input_activations,axis = 3,keepdims = True)

    weights = T.reshape(J*FF_str,[nsam,1,nhid,-1])
    
    hidden_activations = rectify((input_activations*weights).sum(axis = 3) - (TH + (T.sgn(TH_s)*(T.abs_(TH_s)**a))*TH_d))#[nsam,ni,nhid]

    return hidden_activations#/(T.mean(hidden_activations,axis = [1,2]) + .001)
