import pytest

from ..simple_discriminator import _validate_norm, _validate_options


@pytest.mark.parametrize('normalization, n_layers, desired', [
    ('none', 0, ()),
    ('layer', 0, ()),
    ('none', 1, ('none',)),
    ('layer', 1, ('layer',)),
    ('none', 2, ('none', 'none')),
    ('layer', 2, ('layer', 'layer')),
    (['none', 'layer', 'layer'], 3, ['none', 'layer', 'layer']),
])
def test_validate_norm_valid(normalization, n_layers, desired):
    actual = _validate_norm(normalization, n_layers)
    assert actual == desired


@pytest.mark.parametrize('normalization, n_layers', [
    ('spam', 0),
    ((), 1,),
    (('none', 'none'), 1,),
    (('none', 'spam'), 2,),
])
def test_validate_norm_invalid(normalization, n_layers):
    with pytest.raises(AssertionError):
        _validate_norm(normalization, n_layers)


@pytest.mark.parametrize('options, normalization, desired', [
    (None, ('none', 'none'), [{}, {}]),
    ([{'a': 1}, {'b': 2}, {'c': 3}], ('none', 'none', 'none'),
     [{'a': 1}, {'b': 2}, {'c': 3}]),
    ({'none': {'a': 1}, 'layer': {'b': 2}}, ('none', 'layer', 'layer'),
     [{'a': 1}, {'b': 2}, {'b': 2}]),
])
def test_validate_options_valid(options, normalization, desired):
    actual = _validate_options(options, normalization)
    assert actual == desired


@pytest.mark.parametrize('options, normalization', [
    ({'spam': 1}, ()),          # invalid key
    ([{}], ('none', 'none')),   # unmatched length
])
def test_validate_options_invalid(options, normalization):
    with pytest.raises(AssertionError):
        _validate_options(options, normalization)
