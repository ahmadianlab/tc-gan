import lasagne.layers as L

def make_net(input_variable,in_shape):

    net = L.InputLayer(in_shape,input_variable)
    
    net = L.DenseLayer(net,1)

    return net
