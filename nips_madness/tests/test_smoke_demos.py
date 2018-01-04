import importlib

from matplotlib import pyplot
import pytest

from .. import demos


@pytest.mark.parametrize('module, args', [
    ('plot_theano_euler_trajectory', []),
    ('plot_theano_euler_tunings', ['--batchsize', '1']),
])
def test_smoke_slowtest(module, args):
    demo = importlib.import_module('.' + module, demos.__name__)
    demo.main(args)
    pyplot.close('all')


@pytest.mark.parametrize('module, args', [
    ('plot_ssnode_trajectory', ['--plot-fp']),
])
def test_smoke(module, args):
    test_smoke_slowtest(module, args)
