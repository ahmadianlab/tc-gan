import pytest

from ..core import BaseComponent, consume_subdict


class SimpleComponent(BaseComponent):
    def __init__(self, alpha, beta, gamma=3):
        self.alpha = alpha
        self.beta = beta


def test_consume_kwargs_only_kwargs():
    obj, rest = SimpleComponent.consume_kwargs(alpha=1, beta=2, delta=4)
    assert obj.alpha == 1
    assert obj.beta == 2
    assert rest == dict(delta=4)


def test_consume_kwargs_with_posargs():
    obj, rest = SimpleComponent.consume_kwargs(1, 2)
    assert obj.alpha == 1
    assert obj.beta == 2
    assert rest == {}


def test_consume_config_multi_value_error():
    with pytest.raises(ValueError):
        obj, rest = SimpleComponent.consume_config(dict(alpha=1), alpha=2)


def test_consume_config_not_consumed():
    with pytest.raises(ValueError):
        obj, rest = SimpleComponent.consume_config(dict(alpha=1),
                                                   beta=2, delta=4)


def test_consume_subdict_subrest():
    obj, rest = consume_subdict(
        SimpleComponent, 'subkey',
        dict(subkey=dict(alpha=1, beta=2, delta=4), alpha=10, beta=20))
    assert obj.alpha == 1
    assert obj.beta == 2
    assert rest == dict(subkey=dict(delta=4), alpha=10, beta=20)


def test_consume_subdict_remove_empty_subdict():
    obj, rest = consume_subdict(
        SimpleComponent, 'subkey',
        dict(subkey=dict(alpha=1, beta=2), alpha=10, beta=20))
    assert obj.alpha == 1
    assert obj.beta == 2
    assert rest == dict(alpha=10, beta=20)


def test_consume_subdict_missing_subdict():
    obj, rest = consume_subdict(
        SimpleComponent, 'subkey',
        dict(alpha=10, beta=20),
        alpha=1, beta=2)
    assert obj.alpha == 1
    assert obj.beta == 2
    assert rest == dict(alpha=10, beta=20)
