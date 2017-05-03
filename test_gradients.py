import ssnode as SS
import gradient_expressions.SS_grad as grad
import theano
import theano.tensor as T
import lasagne as L
import numpy as np

    #theano shared variables to define the gradient expression

J00 = theano.shared(.5)
J01 = theano.shared(-1.1)
J10 = theano.shared(1.3)
J11 = theano.shared(-.1)

D00 = theano.shared(.1)
D01 = theano.shared(-.1)
D10 = theano.shared(.1)
D11 = theano.shared(-.1)

S00 = theano.shared(5.)
S01 = theano.shared(5.)
S10 = theano.shared(5.)
S11 = theano.shared(5.)

all_params = [J00,J01,J10,J11,D00,D01,D10,D11,S00,S01,S10,S11]

    #a function to pull the parameter values out of the shared variables
def get_params():
    J = [[J00.get_value(),J01.get_value()],[J10.get_value(),J11.get_value()]]
    D = [[D00.get_value(),D01.get_value()],[D10.get_value(),D11.get_value()]]
    S = [[S00.get_value(),S01.get_value()],[S10.get_value(),S11.get_value()]]
    
    return J,D,S


    #parameters to define the weight function

N = 100
Z = np.random.rand(2*N,2*N)

J,delta,sigma = get_params()
ww = grad.generate_weight(N, J, delta, sigma, Z)

print(ww.shape)

    #random rates and inputs, just to test that it runs
ii = np.random.rand(2*N)
nn = 2.
kk = 0.04

#theano variables for gradient expression inputs
R = T.vector("r","float32")
W = T.matrix("w","float32")
DW = T.tensor3("dw","float32")
I = T.vector("i","float32")
n = T.scalar("n","float32")
k = T.scalar("k","float32")

#now in each update we need to generate the weight gradients, then the gradient expressions w.r.t. the shared variables,

WG = grad.WRgrad(R,W,DW,I,n,k)

##this is our stand in for the discriminator! It just needs to give a scalar loss from a fake ss rate

loss = ((R - 1.)*(R - 1.)).sum()

lgrad = T.grad(loss,R)

###############################################

WG = T.tensordot(WG,lgrad,axes = [1,0])

UP = L.updates.nesterov_momentum(WG,all_params,.01)

OF = theano.function([R,W,DW,I,n,k],R,updates = UP,allow_input_downcast = True,on_unused_input = 'ignore')


for epoch in range(100):
    grads = grad.generate_weight_grads(N,J,delta,sigma,Z)

    J,delta,sigma = get_params()
    ww = grad.generate_weight(N, J, delta, sigma, Z)
    
    rr = SS.fixed_point(ww,ii).x

    print(OF(rr,ww,grads,ii,n,k))
    
