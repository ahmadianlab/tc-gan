import pytest

from . import test_wgan
from . import test_cwgan

factory = dict(
    wgan=test_wgan.emit_gan,
    cwgan=test_cwgan.emit_gan,
)


def parametrize_consume_union_tests(f):
    f = pytest.mark.parametrize('gan_type', sorted(factory))(f)
    f = pytest.mark.parametrize('config', [
        dict(V=0.1),
        dict(V_min=0.1),
        dict(V=0.1, V_max=0.1),
    ])(f)
    return f


@parametrize_consume_union_tests
def test_consume_union_true(gan_type, config):
    unconsumed = ['V', 'V_min', 'V_max']
    config = dict(config, consume_union=True)
    _gan, rest = factory[gan_type](**config)
    assert not set(rest) & set(unconsumed)


@parametrize_consume_union_tests
def test_consume_union_false(gan_type, config):
    unconsumed = ['V', 'V_min', 'V_max']
    desired = {k: config[k] for k in unconsumed if k in config}
    config = dict(config, consume_union=False)
    _gan, rest = factory[gan_type](**config)
    actual = {k: rest[k] for k in set(rest) & set(desired)}
    assert actual == desired
