import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne


class LayerNormLayer(lasagne.layers.BatchNormLayer):

    """
    Implementation of Layer Normalization (Ba, Kiros & Hinton, 2016).

    This normalizes input so that it has zero mean and unit variance
    over neurons (as opposed to over batches as in the batch
    normalization).  Since this layer do not have learnable
    parameters, it must be sandwiched by `DenseLayer` and `BiasLayer`
    etc.  See `layer_normalized_dense_layer`.

    The current implementation assumes that the first (0th) axis is
    the batch dimension and other dimensions are used to calculate the
    mean and variance.  In particular, it does not support recurrent
    layers.

    - Ba, Kiros & Hinton (2016) "Layer Normalization."
      http://arxiv.org/abs/1607.06450
    - https://github.com/Lasagne/Lasagne/issues/736#issuecomment-241374360

    """

    def __init__(self, incoming, axes='auto', **kwargs):
        if axes != 'auto':
            kwargs['axes'] = axes

        super(LayerNormLayer, self).__init__(
            incoming,
            beta=None, gamma=None,
            **kwargs)

        if axes == 'auto':
            self.axes = tuple(range(1, len(self.input_shape)))

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
                                 W=lasagne.init.Normal(std=1),
                                 b=lasagne.init.Constant(0.),
                                 use_scale='auto',
                                 **kwargs):
    assert num_units > 1

    assert use_scale in (True, False, 'auto')
    if use_scale == 'auto':
        use_scale = nonlinearity is not NL.rectify

    dense_kwargs = {}
    for key in {'num_leading_axes'} & set(kwargs):
        dense_kwargs[key] = kwargs.pop(key)
    layer = L.DenseLayer(incoming, num_units,
                         W=W,
                         b=None,
                         nonlinearity=NL.linear,
                         **dense_kwargs)
    layer = LayerNormLayer(layer, **kwargs)
    if use_scale:
        layer = L.ScaleLayer(layer)
    layer = L.BiasLayer(layer, b=b)
    return L.NonlinearityLayer(layer, nonlinearity=nonlinearity)


normalization_types = {
    'none': L.DenseLayer,
    'layer': layer_normalized_dense_layer,
}


def _validate_norm(normalization, n_layers):
    if isinstance(normalization, (list, tuple)):
        assert len(normalization) == n_layers
        assert all(n in normalization_types for n in normalization)
        return normalization
    else:
        assert normalization in normalization_types
        return (normalization,) * n_layers


def _validate_options(options, normalization):
    if options is None:
        options = [{}] * len(normalization)
    elif isinstance(options, dict):
        assert set(options) <= set(normalization_types)
        options = [options.get(key, {}) for key in normalization]
    assert len(options) == len(normalization)
    return options


def make_net(in_shape, loss_type, layers=[], normalization='none',
             nonlinearity='rectify', options=None):
    """
    Make a discriminator network appropriate for loss `LOSS`.

    Parameters
    ----------
    in_shape : tuple of int
        `shape` argument passed to `lasagne.layers.InputLayer`.
    loss_type : {'LS', 'CE', 'WGAN', 'WD'}
        Discriminator loss type which is used to determine the output
        layer and nonlinearity type.
        For backward compatibility, 'WGAN' means 'WD'.
    layers : tuple/list of int
        Numbers of units in hidden layers. Empty tuple means perceptron.
    normalization : {'none', 'layer'} or list of them
        (1) If it is a *simple normalization specification* such as
        ``none`` and ``layer``, it specifies the normalization of the
        hidden layers.
        (2) It can be a tuple or list of simple normalization
        specifications to specify normalization for each layer
        explicitly.  Its length has to be equal to ``len(layers)``.
    nonlinearity : str or callable
        Nonlinearity to be used for *hidden* layers.  A string for
        specifying a function in `lasagne.nonlinearities` or any
        callable.

    """

    net = L.InputLayer(in_shape)
    net = stack_hidden_layers(net, layers, normalization, nonlinearity,
                              options)
    return make_output_layer(net, loss_type)


def stack_hidden_layers(net, layers, normalization, nonlinearity,
                        options=None):

    if isinstance(nonlinearity, str):
        nonlinearity = getattr(NL, nonlinearity)
    normalization = _validate_norm(normalization, len(layers))
    options = _validate_options(options, normalization)

    for width, layer_type, opt in zip(layers, normalization, options):
        make_layer = normalization_types[layer_type]
        net = make_layer(net, width, b=lasagne.init.Normal(.01, 0),
                         nonlinearity=nonlinearity, **opt)

    return net


def make_output_layer(net, loss_type):
    if loss_type == "LS":
        net = L.DenseLayer(net,1,nonlinearity = NL.linear,b=lasagne.init.Normal(.01,0))
    elif loss_type == "CE":
        net = L.DenseLayer(net,1,nonlinearity = NL.sigmoid,b=lasagne.init.Normal(.01,0))
    elif loss_type in ("WD", "WGAN"):
        net = L.DenseLayer(net, 1, nonlinearity=NL.linear, b=None)
    else:
        raise ValueError("Invaid loss_type specified: {}".format(loss_type))

    return net
