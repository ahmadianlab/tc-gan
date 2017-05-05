import lasagne.layers as L
import lasagne.nonlinearities as NL

def make_net(input_variable,in_shape,params = None):

    net = L.InputLayer(in_shape,input_variable)

    if params == None:        
                
        net = L.DenseLayer(net,1,nonlinearity = NL.sigmoid)

    else:
        
        net = L.DenseLayer(net,1,nonlinearity = NL.sigmoid,W = params[1].W,b = params[1].b)        
 
    return net
