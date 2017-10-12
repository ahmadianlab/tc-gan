from unittest import mock

import numpy as np
import pytest

from .. import recorders
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
        rec.record(*[0] * len(recorders.DiscLearningRecorder.column_names))


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


@pytest.mark.parametrize('recclass', sorted(recorder_faker_map, key=str))
def test_record_shape(recclass):
    driver = fake_driver()
    rec = recclass.from_driver(driver)

    faker = recorder_faker_map[recclass]()
    faker.send_record(rec)
    faker.send_record(rec)
    faker.send_record(rec)

    loaded = load_csv(driver.datastore, rec.filename)
    assert loaded.shape == (3, len(rec.column_names))


def test_generator_param_order():
    recclass = recorders.GenParamRecorder
    driver = fake_driver()
    rec = recclass.from_driver(driver)

    faker = recorder_faker_map[recclass]()
    faker.send_record(rec)

    J = driver.gan.J.get_value.return_value
    D = driver.gan.D.get_value.return_value
    S = driver.gan.S.get_value.return_value
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

    driver.datastore.tables.saverow.assert_called_with(
        recclass.filename, [0] + flat_JDS, echo=False)
    assert len(flat_JDS) == len(set(flat_JDS))
