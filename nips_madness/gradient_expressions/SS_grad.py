import theano
import theano.tensor as T
import lasagne as L
import numpy as np

from ..weight_gen import weight, generate_weight

#what I need is: to write an expression that takes a variable R and matrix W (ss response and weight matrix) and returns the gradient.
#Now I need functions that take the parameters and compute Wgrads w.r.t. the parameters

def weight_DJ(x, J, delta, sigma, z):
    '''
    
    gradients w.r.t. J

    '''
    return np.exp(-(x - x.T)**2/(2 * sigma**2)) * J

def weight_DS(x, J, delta, sigma, z):
    '''
    
    gradients w.r.t. S

    '''
    return 2*((x - x.T)**2/(2 * sigma**3))*np.exp(-(x - x.T)**2/(2 * sigma**2)) * (J + delta * z)

def weight_DD(x, J, delta, sigma, z):
    '''
    
    gradients w.r.t. D

    '''
    return np.exp(-(x - x.T)**2/(2 * sigma**2)) * z

def generate_weight_grads(N, J, delta, sigma, z):

    '''

    Returns [Dw/Dth  for th in J,D,S]
 
    This returns a tensor of dims [12,2N,2N] where each entry along the first axis is dW/dth w.r.t. a single th. The th are arranged as follows:

    0 - 3  : J
    4 - 7  : D
    8 - 11 : S

    '''

    # Copy J and delta and turn them into inhibition-aware matrices:
    J = np.array(J) # copying since we are modifying it in-place
    J[:, 1] *= -1
    delta = np.array(delta) # ditto
    delta[:, 1] *= -1

    sigma = np.asarray(sigma)
    x = np.linspace(-0.5, 0.5, N).reshape((1, -1))
    W = np.empty((12, 2 * N, 2 * N))

    W[0,:N, :N] = weight_DJ(x, J[0, 0], delta[0, 0], sigma[0, 0], z[:N,:N])
    W[1,N:, :N] = weight_DJ(x, J[1, 0], delta[1, 0], sigma[1, 0], z[N:,:N])
    W[2,:N, N:] = weight_DJ(x, J[0, 1], delta[0, 1], sigma[0, 1], z[:N,N:])
    W[3,N:, N:] = weight_DJ(x, J[1, 1], delta[1, 1], sigma[1, 1], z[N:,N:])

    W[4,:N, :N] = weight_DD(x, J[0, 0], delta[0, 0], sigma[0, 0], z[:N,:N])
    W[5,N:, :N] = weight_DD(x, J[1, 0], delta[1, 0], sigma[1, 0], z[N:,:N])
    W[6,:N, N:] = weight_DD(x, J[0, 1], delta[0, 1], sigma[0, 1], z[:N,N:])
    W[7,N:, N:] = weight_DD(x, J[1, 1], delta[1, 1], sigma[1, 1], z[N:,N:])

    W[8,:N, :N] = weight_DS(x, J[0, 0], delta[0, 0], sigma[0, 0], z[:N,:N])
    W[9,N:, :N] = weight_DS(x, J[1, 0], delta[1, 0], sigma[1, 0], z[N:,:N])
    W[10,:N, N:] = weight_DS(x, J[0, 1], delta[0, 1], sigma[0, 1], z[:N,N:])
    W[11,N:, N:] = weight_DS(x, J[1, 1], delta[1, 1], sigma[1, 1], z[N:,N:])

    return W


def WRgrad(R,W,DW,I,n,k):

    '''

    This expression takes theano variables for R,W,dW/dth,I,n, and k and outputs the gradient dr/dth

    '''
   
    V = T.dot(W,R) + I
 
    phi = T.reshape(n*k*T.power(V,n-1),[-1,1])

    J = T.identity_like(W) - phi*W

    B = T.reshape(phi,[-1])*T.dot(DW,R)
    
    return T.dot(T.nlinalg.MatrixInverse()(J),B)

if __name__ == "__main__":

    #theano shared variables to define the gradient expression

    J00 = theano.shared(.1,name = 'j00')
    J01 = theano.shared(1.,name = 'j01')
    J10 = theano.shared(1.,name = 'j10')
    J11 = theano.shared(1.,name = 'j11')

    D00 = theano.shared(1.,name = 'd00')
    D01 = theano.shared(1.,name = 'd01')
    D10 = theano.shared(1.,name = 'd10')
    D11 = theano.shared(1.,name = 'd11')
    
    S00 = theano.shared(1.,name = 's00')
    S01 = theano.shared(1.,name = 's01')
    S10 = theano.shared(1.,name = 's10')
    S11 = theano.shared(1.,name = 's11')

    all_params = [J00,J01,J10,J11,D00,D01,D10,D11,S00,S01,S10,S11]

    #a function to pull the parameter values out of the shared variables
    def get_params():
        J = [[J00.get_value(),J01.get_value()],[J10.get_value(),J11.get_value()]]
        D = [[D00.get_value(),D01.get_value()],[D10.get_value(),D11.get_value()]]
        S = [[S00.get_value(),S01.get_value()],[S10.get_value(),S11.get_value()]]

        return J,D,S

    J,delta,sigma = get_params()

    #parameters to define the weight function
 
    N = 20
    Z = np.random.rand(2*N,2*N)

    ww = generate_weight(N, J, delta, sigma, Z)

    print(ww.shape)

    #random rates and inputs, just to test that it runs
    rr = np.random.rand(2*N)
    ii = np.random.rand(2*N)
    nn = 2.2
    kk = 0.04

    #theano variables for gradient expression inputs
    R = T.vector("r","float32")
    W = T.matrix("w","float32")
    DW = [T.matrix("dw_" + p.name,"float32") for p in all_params]
    I = T.vector("i","float32")
    n = T.scalar("n","float32")
    k = T.scalar("k","float32")

    #now in each update we need to generate the weight gradients, then the gradient expressions w.r.t. the shared variables,
    
    grads = generate_weight_grads(N,J,delta,sigma,Z)

    for g in grads:
        print("dloop: {}".format(g.mean()))

    WG = [WRgrad(R,W,d,I,n,k) for d in DW]

    ##this is our stand in for the discriminator! It just needs to give a scalar loss from a fake ss rate

    loss = (R*R).sum()
   
    lgrad = T.grad(loss,R)

    ##

    WG = [T.dot(g,lgrad) for g in WG]
                       
    OF = theano.function([R,W,I,n,k] + DW,WG,allow_input_downcast = True,on_unused_input = 'ignore')

    print(np.asarray(OF(rr,ww,ii,nn,kk,grads[0],grads[1],grads[2],grads[3],grads[4],grads[5],grads[6],grads[7],grads[8],grads[9],grads[10],grads[11])).mean())
    
