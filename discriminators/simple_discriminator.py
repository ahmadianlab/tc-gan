import lasagne.layers as L
import lasagne.nonlinearities as NL

def make_net(input_variable,in_shape,LOSS,layers = [],params = None):

    net = L.InputLayer(in_shape,input_variable)

    if params == None:        

        for l in range(len(layers)):
            net = L.DenseLayer(net,layers[l])
                
        if LOSS == "LS":
            net = L.DenseLayer(net,1,nonlinearity = NL.linear)
        elif LOSS == "CE":
            net = L.DenseLayer(net,1,nonlinearity = NL.sigmoid)
        elif LOSS == "WGAN":
            net = L.DenseLayer(net,1,nonlinearity = NL.linear)
        else:
            print("Invaid LOSS specified")
            exit()

    else:
        for l in range(len(layers)):
            net = L.DenseLayer(net,layers[l],W = params[l+1].W,b = params[l+1].b)
        
        if LOSS == "LS":
            net = L.DenseLayer(net,1,nonlinearity = NL.linear,W = params[len(layers)+1].W,b = params[len(layers)+1].b)
        elif LOSS == "CE":
            net = L.DenseLayer(net,1,nonlinearity = NL.sigmoid,W = params[len(layers)+1].W,b = params[len(layers)+1].b)
        elif LOSS == "WGAN":
            net = L.DenseLayer(net,1,nonlinearity = NL.linear,W = params[len(layers)+1].W,b = params[len(layers)+1].b)
        else:
            print("Invaid LOSS specified")
            exit()
 
    return net
