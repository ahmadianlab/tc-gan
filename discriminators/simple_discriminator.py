import lasagne.layers as L
import lasagne.nonlinearities as NL

def make_net(input_variable,in_shape,layers = [],params = None):

    net = L.InputLayer(in_shape,input_variable)

    if params == None:        

        for l in range(len(layers)):
            net = L.DenseLayer(net,layers[l])
                
        net = L.DenseLayer(net,1,nonlinearity = NL.sigmoid)


    else:
        for l in range(len(layers)):
            net = L.DenseLayer(net,layers[l],W = params[l+1].W,b = params[l+1].b)
        
        net = L.DenseLayer(net,1,nonlinearity = NL.sigmoid,W = params[len(layers)+1].W,b = params[len(layers)+1].b)        
 
    return net
