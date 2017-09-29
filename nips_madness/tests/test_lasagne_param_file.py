import lasagne
import numpy

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
