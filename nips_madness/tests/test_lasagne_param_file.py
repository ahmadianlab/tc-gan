import lasagne
import numpy
import pytest

from .. import lasagne_param_file


def test_dump_then_load(tmpdir):
    l0 = lasagne.layers.InputLayer((2, 3))
    l1 = lasagne.layers.DenseLayer(l0, 5)
    param_path = str(tmpdir.join('param_file.npz'))
    lasagne_param_file.dump(l1, param_path)
    values = lasagne_param_file.load(param_path)
    assert isinstance(values, list)
    desired = lasagne.layers.get_all_param_values(l1)
    numpy.testing.assert_equal(values, desired)


def test_save_on_error(tmpdir):
    l0 = lasagne.layers.InputLayer((2, 3))
    l1 = lasagne.layers.DenseLayer(l0, 5)
    pre_path = str(tmpdir.join('pre_error.npz'))
    post_path = str(tmpdir.join('post_error.npz'))

    values = [arr.copy() for arr in lasagne.layers.get_all_param_values(l1)]

    class MyException(Exception):
        pass

    try:
        with lasagne_param_file.save_on_error(l1, pre_path, post_path):
            newvalues = [arr + 1 for arr in values]
            lasagne.layers.set_all_param_values(l1, newvalues)
            raise MyException
    except MyException:
        pass

    stored = lasagne_param_file.load(pre_path)
    numpy.testing.assert_equal(stored, values)
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(stored, newvalues)

    stored = lasagne_param_file.load(post_path)
    numpy.testing.assert_equal(stored, newvalues)
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(stored, values)
