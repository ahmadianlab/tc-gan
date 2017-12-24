from unittest import mock

import numpy as np
import pytest

from .. import recorders
from ..networks.simple_discriminator import make_net
from ..networks.tests.test_tuning_curve import emit_tcg_for_test
from .test_drivers import fake_datastore, setup_fake_gan, \
    GenFakeUpdateResults


def fake_driver():
    driver = mock.Mock()
    driver.datastore = fake_datastore()
    setup_fake_gan(driver.gan)
    return driver


def load_csv(datastore, name, skiprows=1, **kwargs):
    stream, = datastore.path.side_effect.returned[name, ]
    stream.seek(0)
    loaded = np.loadtxt(stream, delimiter=',', skiprows=skiprows, **kwargs)
    if loaded.ndim == 1:
        loaded = loaded.reshape((1, -1))
    return loaded


class FakeLearningRecords(object):

    def __init__(self):
        self.results = GenFakeUpdateResults()

    def send_record(self, rec):
        rec.record(0, self.results.new())


class FakeDiscLearningRecords(object):

    def send_record(self, rec):
        rec.record(*[0] * len(rec.column_names))


class FakeGenParamRecords(object):

    def send_record(self, rec):
        rec.record(0)


class FakeDiscParamStatsRecords(object):

    def send_record(self, rec):
        rec.record(0, 0)


recorder_faker_map = {
    recorders.LearningRecorder: FakeLearningRecords,
    recorders.DiscLearningRecorder: FakeDiscLearningRecords,
    recorders.GenParamRecorder: FakeGenParamRecords,
    recorders.DiscParamStatsRecorder: FakeDiscParamStatsRecords,
}


def setup_recorder(recclass):
    driver = mock.Mock()
    driver.gan.gen, _ = gen, _ = emit_tcg_for_test()

    if recclass is recorders.GenParamRecorder:
        driver.gan.get_gen_param.return_value \
            = [p.get_value() for p in gen.get_all_params()]
    elif recclass is recorders.DiscParamStatsRecorder:
        layers = [16, 16]
        normalization = ['none', 'layer']
        driver.gan.discriminator \
            = make_net((2, 3), 'WGAN', layers, normalization)

    rec = recclass.from_driver(driver)
    rec._saverow = mock.Mock()

    return rec, driver


@pytest.mark.parametrize('recclass', sorted(recorder_faker_map, key=str))
def test_record_shape(recclass):
    rec, _ = setup_recorder(recclass)

    faker = recorder_faker_map[recclass]()
    faker.send_record(rec)
    faker.send_record(rec)
    faker.send_record(rec)

    saverow = rec._saverow
    assert len(saverow.call_args_list) == 3
    lengths = [len(row) for (row,), _kwds in saverow.call_args_list]
    assert set(lengths) == {len(rec.column_names)}


@pytest.mark.parametrize('recclass', [
    recorders.GenParamRecorder,
    recorders.FlexGenParamRecorder,
])
def test_generator_param_order(recclass):
    rec, driver = setup_recorder(recclass)

    gen_step = 0
    rec.record(gen_step)

    gen = driver.gan.gen
    J, D, S = [p.get_value() for p in gen.get_all_params()]
    E = 0
    I = 1
    flat_JDS = [
        # J
        J[E, E], J[E, I],
        J[I, E], J[I, I],
        # D (\Delta)
        D[E, E], D[E, I],
        D[I, E], D[I, I],
        # S (\sigma)
        S[E, E], S[E, I],
        S[I, E], S[I, I],
    ]

    saverow = rec._saverow
    saverow.assert_called_with([gen_step] + flat_JDS)
    assert len(set(flat_JDS)) > 1  # make sure the test was non-trivial

    if recclass is recorders.GenParamRecorder:
        driver.gan.get_gen_param.assert_called_once()


def test_paramstats_layernorm():
    recclass = recorders.DiscParamStatsRecorder
    driver = fake_driver()
    layers = [16, 16]
    normalization = ['none', 'layer']
    driver.gan.discriminator = make_net((2, 3), 'WGAN', layers, normalization)
    rec = recclass.from_driver(driver)
    rec.record(0, 0)
