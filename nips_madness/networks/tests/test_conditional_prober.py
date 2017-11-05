from unittest import mock

import numpy as np
import theano

from ..cwgan import ConditionalProber
from .test_cwgan import make_gan


def mock_model():
    model = mock.Mock()
    model.num_sites = 201
    model.time_avg = theano.tensor.tensor3('time_avg')
    model.stimulator.contrasts = theano.tensor.matrix('contrasts')
    return model


def test_conditional_probes():
    prober = ConditionalProber(mock_model())

    norm_probes = np.array([-1, -0.5, 0, 0.5, 1] * 2,
                           dtype=theano.config.floatX)
    cell_types = np.array([0] * 5 + [1] * 5, dtype='uint16')

    probes = prober.probes.eval({
        prober.norm_probes: norm_probes,
        prober.cell_types: cell_types,
    })
    desired = [0, 50, 100, 150, 200,
               201, 251, 301, 351, 401]
    np.testing.assert_equal(probes, desired)


def test_conditional_probes_compare_with_sample_sites():
    gan, _ = make_gan()
    norm_probes = gan.norm_probes
    cell_types = np.zeros_like(norm_probes)
    probes = gan.gen.prober.eval.probes(
        norm_probes=norm_probes,
        cell_types=cell_types,
    )
    sample_sites = list(gan.sample_sites)
    probes = list(probes)
    assert sample_sites == probes


def test_conditional_tuning_curve():
    prober = ConditionalProber(mock_model())

    num_models = 2
    num_tcdom = 3
    num_neurons = prober.model.num_sites * 2
    norm_probes = np.array([-1, -0.5, 0, 0.5, 1] * 2,
                           dtype=theano.config.floatX)
    cell_types = np.array([0] * 5 + [1] * 5, dtype='uint16')
    model_ids = np.array([0, 1] * 5, dtype='uint16')
    shape = (num_models, num_tcdom, num_neurons)
    time_avg = np.arange(np.prod(shape),
                         dtype=theano.config.floatX).reshape(shape)

    assert norm_probes.shape == cell_types.shape == model_ids.shape
    assert all(model_ids < num_models)

    tuning_curve = prober.tuning_curve.eval({
        prober.norm_probes: norm_probes,
        prober.cell_types: cell_types,
        prober.model_ids: model_ids,
        prober.model.time_avg: time_avg,
    })
    assert tuning_curve.shape == (len(norm_probes), num_tcdom)

    # Following `desired` value was copied from `tuning_curve` to
    # "quench" the implementation.
    desired = [
        [0,      402,   804],
        [1256,  1658,  2060],
        [100,    502,   904],
        [1356,  1758,  2160],
        [200,    602,  1004],
        [1407,  1809,  2211],
        [251,    653,  1055],
        [1507,  1909,  2311],
        [351,    753,  1155],
        [1607,  2009,  2411],
    ]
    np.testing.assert_equal(tuning_curve, desired)
