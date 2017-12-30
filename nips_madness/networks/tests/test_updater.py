from types import SimpleNamespace

import numpy as np
import theano

from ..wgan import Updater


def update_single_param(updater, before):
    param = theano.shared(np.asarray(before, dtype=theano.config.floatX))
    loss = param.mean() * 0  # multiply by 0 to ignore the loss part
    updates = updater(loss, [param])
    fun = theano.function([], loss, updates=updates, mode='FAST_COMPILE')
    fun()

    return SimpleNamespace(
        loss=loss,
        updates=updates,
        param=param,
        after=param.get_value(),
    )


def test_l2_decay_no_loss():
    before = np.array([1, 2, 3])
    reg_l2_decay = 0.2
    updater = Updater(learning_rate=1, reg_l2_decay=reg_l2_decay)
    results = update_single_param(updater, before)
    desired = before * (1 - reg_l2_decay)
    np.testing.assert_allclose(results.after, desired)


def test_l1_decay_no_loss():
    before = np.array([1, -1])
    reg_l1_decay = 0.5
    updater = Updater(learning_rate=1, reg_l1_decay=reg_l1_decay)
    results = update_single_param(updater, before)
    desired = [0.5, -0.5]
    np.testing.assert_allclose(results.after, desired)
