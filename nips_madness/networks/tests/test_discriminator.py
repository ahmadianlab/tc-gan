import pytest

from ..simple_discriminator import normalization_types, _validate_norm

valid_norm = 'none'
invalid_norm = 'spam'
assert valid_norm in normalization_types
assert invalid_norm not in normalization_types


@pytest.mark.parametrize('norm, n_layers, desired', [
    (valid_norm, 0, []),
    ('layer', 0, []),
    (valid_norm, 3, [valid_norm] * 3),
    ((valid_norm,), 1, [valid_norm]),
    ((valid_norm,) * 5, 5, [valid_norm] * 5),
    (('none', 'layer', 'layer'), 3, ['none', 'layer', 'layer']),
])
def test_validate_norm_valid(norm, n_layers, desired):
    actual = _validate_norm(norm, n_layers)
    assert list(actual) == desired


@pytest.mark.parametrize('norm, n_layers', [
    (invalid_norm, 0),
    ((invalid_norm,), 1),
    ((valid_norm,), 3),
])
def test_validate_norm_invalid(norm, n_layers):
    with pytest.raises(AssertionError):
        _validate_norm(norm, n_layers)
