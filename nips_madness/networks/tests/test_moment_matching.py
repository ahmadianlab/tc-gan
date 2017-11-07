import numpy as np
import pytest

from ..moment_matching import BPTTMomentMatcher, MOMENT_WEIGHT_TYPES


def calc_moment_weights(data, type='mean', regularization=1e-3, lam=1):
    class CustomMomentMatcher(BPTTMomentMatcher):

        def __init__(self):
            self.lam = lam
            self.moment_weights_regularization = regularization
            self.moment_weight_type = type

    self = CustomMomentMatcher()
    self.set_dataset(data)
    return self.moment_weights


@pytest.mark.parametrize('type', MOMENT_WEIGHT_TYPES)
@pytest.mark.parametrize('num_tcdom', [1, 2, 5])
def test_moment_weights_shape(type, num_tcdom):
    moment_weights = calc_moment_weights(np.ones((3, num_tcdom)), type)
    assert moment_weights.shape == (2, num_tcdom)


@pytest.mark.parametrize('type', MOMENT_WEIGHT_TYPES)
@pytest.mark.parametrize('num_tcdom', [1, 2, 5])
def test_moment_weights_lam0(type, num_tcdom):
    wm, wv = calc_moment_weights(np.ones((3, num_tcdom)), type, lam=0)
    assert (wm > 0).any()
    assert (wv == 0).all()
