import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne


class LayerNormLayer(lasagne.layers.BatchNormLayer):

    """
    Implementation of Layer Normalization (Ba, Kiros & Hinton, 2016).

    This layer normalizes input so that it has zero mean and unit
    variance over neurons (as opposed to over batches as in the batch
    normalization).  Since this layer do not have learnable
    parameters, it must be sandwiched by `DenseLayer` and `BiasLayer`
    etc.  See `layer_normalized_dense_layer`.

    - Ba, Kiros & Hinton (2016) "Layer Normalization."
      http://arxiv.org/abs/1607.06450
    - https://github.com/Lasagne/Lasagne/issues/736#issuecomment-241374360

    """

    def __init__(self, incoming, axes=-1, **kwargs):
        super(LayerNormLayer, self).__init__(incoming, axes=axes, **kwargs)

    def get_output_for(self, input,
                       batch_norm_use_averages=False,
                       batch_norm_update_averages=False,
                       **kwargs):
        return super(LayerNormLayer, self).get_output_for(
            input,
            batch_norm_use_averages=batch_norm_use_averages,
            batch_norm_update_averages=batch_norm_update_averages,
            **kwargs)


def layer_normalized_dense_layer(incoming, num_units,
                                 nonlinearity=NL.rectify,
                                 b=lasagne.init.Constant(0.),
                                 **kwargs):
    layer = L.DenseLayer(incoming, num_units, b=None, **kwargs)
    layer = LayerNormLayer(layer)
    layer = L.ScaleLayer(layer)
    layer = L.BiasLayer(layer, b=b)
    return L.NonlinearityLayer(layer)


def make_net(in_shape, LOSS, layers=[], normalization=None):

    net = L.InputLayer(in_shape)

    assert normalization in (None, 'layer')
    if normalization is None:
        make_layer = L.DenseLayer
    else:
        make_layer = layer_normalized_dense_layer

    for l in range(len(layers)):
        net = make_layer(net,layers[l],b=lasagne.init.Normal(.01,0))

    if LOSS == "LS":
        net = make_layer(net,1,nonlinearity = NL.linear,b=lasagne.init.Normal(.01,0))
    elif LOSS == "CE":
        net = make_layer(net,1,nonlinearity = NL.sigmoid,b=lasagne.init.Normal(.01,0))
    elif LOSS == "WGAN":
        net = make_layer(net,1,nonlinearity = NL.linear,b=lasagne.init.Normal(.01,0))
    else:
        raise ValueError("Invaid LOSS specified: {}".format(LOSS))

    return net
