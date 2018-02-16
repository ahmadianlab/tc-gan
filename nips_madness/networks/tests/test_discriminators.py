import lasagne
import pytest

from ..cwgan import ConditionalDiscriminator
from ..simple_discriminator import LayerNormLayer
from ..wgan import UnConditionalDiscriminator


def make_ucdisc(
        shape=(2, 2), loss_type='WD', layers=[2, 2],
        normalization='none', nonlinearity='rectify',
        **kwargs):
    return UnConditionalDiscriminator(
        shape=shape, loss_type=loss_type, layers=layers,
        normalization=normalization, nonlinearity=nonlinearity,
        **kwargs)


def make_cdisc(
        shape=(2, 2), cond_shape=(2, 1), loss_type='WD', layers=[2, 2],
        normalization='none', nonlinearity='rectify',
        **kwargs):
    return ConditionalDiscriminator(
        shape=shape, cond_shape=cond_shape, loss_type=loss_type, layers=layers,
        normalization=normalization, nonlinearity=nonlinearity,
        **kwargs)


def get_layer_norm_layers(layer):
    return [l for l in lasagne.layers.get_all_layers(layer)
            if isinstance(l, LayerNormLayer)]


@pytest.mark.parametrize('make_disc', [make_ucdisc, make_cdisc])
def test_disc_layer_epsilon(make_disc):
    epsilon = 0.123456
    disc = make_disc(layers=[2, 3, 4],
                     normalization=['none', 'layer', 'layer'],
                     net_options=dict(layer=dict(epsilon=epsilon)))
    lnls = get_layer_norm_layers(disc.l_out)
    assert len(lnls) == 2
    assert [l.epsilon for l in lnls] == [epsilon, epsilon]
