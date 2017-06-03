import lasagne
import numpy as np
import pytest

from ..simple_discriminator import LayerNormLayer, layer_normalized_dense_layer


def np_norm_layer(x, epsilon=0):
    kwds = dict(axis=tuple(range(1, len(x.shape))), keepdims=True)
    mean = x.mean(**kwds)
    std = np.sqrt(x.var(**kwds) + epsilon)
    return (x - mean) / std


@pytest.mark.parametrize('def_shape, real_shape', [
    ((2, 3), None),
    ((None, 3), (2, 3)),
    ((2, 3, 4), None),
    ((None, 3, 4), (2, 3, 4)),
])
def test_layer_norm_layer(def_shape, real_shape):
    l0 = lasagne.layers.InputLayer(def_shape)
    l1 = LayerNormLayer(l0)
    out = lasagne.layers.get_output(l1)

    rs = np.random.RandomState(0)
    x = rs.randn(*(real_shape or def_shape))
    actual = out.eval({l0.input_var: x})

    desired = np_norm_layer(x, l1.epsilon)
    assert desired.shape == x.shape

    np.testing.assert_almost_equal(actual, desired)


@pytest.mark.parametrize('batchsize, in_dim, out_dim', [
    (2, 3, 4),
    (10, 30, 20),
])
def test_layer_normalized_dense_layer(batchsize, in_dim, out_dim):
    rs = np.random.RandomState(0)
    x = rs.randn(batchsize, in_dim)
    W = rs.randn(in_dim, out_dim)
    b = rs.randn(out_dim)

    l0 = lasagne.layers.InputLayer(x.shape)
    l1 = layer_normalized_dense_layer(l0, out_dim, W=W, b=b)
    for layer in lasagne.layers.get_all_layers(l1):
        if isinstance(layer, LayerNormLayer):
            assert hasattr(layer, 'epsilon')
            layer.epsilon = 0
            break
    else:
        raise ValueError('LayerNormLayer not found')
    out = lasagne.layers.get_output(l1)
    actual = out.eval({l0.input_var: x})

    a = np.tensordot(x, W, axes=1)
    desired = l1.nonlinearity(np_norm_layer(a) + b)

    assert (desired > 0).any()
    np.testing.assert_almost_equal(actual, desired)
