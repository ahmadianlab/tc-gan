from types import SimpleNamespace
from unittest import mock
import collections
import io

import lasagne
import numpy as np
import pytest

from ..run.gan import init_driver, LearningRecorder
from ..ssnode import DEFAULT_PARAMS


class FakeDataStorePath(object):

    def __init__(self):
        self.returned = collections.defaultdict(list)

    def __call__(self, *args):
        bio = io.BytesIO()
        self.returned[args].append(bio)
        return bio


def make_discriminator():
    l0 = lasagne.layers.InputLayer((2, 3))
    l1 = lasagne.layers.DenseLayer(l0, 5)
    return l1


def make_driver(
        iterations,
        quit_JDS_threshold=-1,
        quiet=False,
        disc_param_save_interval=5,
        disc_param_template='last.npz',
        disc_param_save_on_error=False,
        J0=DEFAULT_PARAMS['J'],
        D0=DEFAULT_PARAMS['D'],
        S0=DEFAULT_PARAMS['S'],
        **run_config):

    # Setup Fake datastore
    datastore = mock.Mock()
    datastore.path.side_effect = FakeDataStorePath()

    rc = SimpleNamespace(**init_driver(
        datastore,
        iterations=iterations,
        quiet=quiet,
        disc_param_save_interval=disc_param_save_interval,
        disc_param_template=disc_param_template,
        disc_param_save_on_error=disc_param_save_on_error,
        quit_JDS_threshold=quit_JDS_threshold,
        J0=J0,
        D0=D0,
        S0=S0,
        **run_config))

    # Setup Fake GAN
    rc.gan.discriminator = make_discriminator()

    rc.gan.rate_penalty_func = mock.Mock()
    rc.gan.rate_penalty_func.side_effect = lambda _: np.nan

    rc.gan.get_reduced = mock.Mock()
    rc.gan.get_reduced.side_effect = lambda _: np.array([[0]])

    for name in 'JDS':
        fake_shared = mock.Mock()
        fake_shared.get_value.side_effect \
            = lambda: np.arange(4).reshape((2, 2))
        setattr(rc.gan, name, fake_shared)

    return rc


HistoryRecord = collections.namedtuple('HistoryRecord', ['args', 'result'])


class BaseFakeUpdateResults(object):

    size = None  # to be configured
    rand_max = 100000

    def __init__(self, seed=1):
        self.rng = np.random.RandomState(seed)
        self.history = []

    def make_result(self):
        return self.rng.randint(self.rand_max, size=self.size)

    def new(self, *args):
        result = self.make_result()
        self.history.append(HistoryRecord(args, result))
        return result


class DiscFakeUpdateResults(BaseFakeUpdateResults):
    fields = ('Dloss', 'Daccuracy', 'SSsolve_time', 'gradient_time')
    size = len(fields)


class GenFakeUpdateResults(BaseFakeUpdateResults):
    fields = (
        'Gloss',
        'Dloss',
        'Daccuracy',
        'rtest',
        'true',
        'SSsolve_time',
        'gradient_time',
        'model_info.rejections',
        'model_info.unused',
    )
    size = len(fields)

    def make_result(self):
        result_dict = dict(zip(
            self.fields,
            super(GenFakeUpdateResults, self).make_result(),
        ))
        result_dict['model_info'] = SimpleNamespace(
            rejections=result_dict['model_info.rejections'],
            unused=result_dict['model_info.unused'],
        )
        result = SimpleNamespace(**result_dict)
        result.true = np.array([[result.true]])
        return result


@pytest.mark.parametrize('iterations', range(1, 4))
def test_gan_driver_iterate(iterations):
    rc = make_driver(iterations=iterations)
    driver = rc.driver

    disc_update_results = DiscFakeUpdateResults()
    gen_update_results = GenFakeUpdateResults()
    n_critic = 5

    @driver.iterate
    def update_func(gen_step):
        for disc_step in range(n_critic):
            d_result = disc_update_results.new(gen_step, disc_step)
            driver.post_disc_update(gen_step, disc_step, *d_result)
        return gen_update_results.new(gen_step)

    assert len(disc_update_results.history) == iterations * n_critic
    assert len(gen_update_results.history) == iterations

    assert rc.gan.rate_penalty_func.call_args_list == [
        mock.call(rec.result.rtest)
        for rec in gen_update_results.history
    ]

    assert rc.gan.get_reduced.call_args_list == [
        mock.call(rec.result.rtest)
        for rec in gen_update_results.history
    ]

    empty_calls = [mock.call()] * iterations
    assert rc.gan.J.get_value.call_args_list == empty_calls
    assert rc.gan.D.get_value.call_args_list == empty_calls
    assert rc.gan.S.get_value.call_args_list == empty_calls

    rc.datastore.path.assert_called_with('disc_param', 'last.npz')

    # Make sure headers (column names) are saved:
    rc.datastore.tables.saverow.assert_any_call(
        LearningRecorder.filename,
        LearningRecorder.column_names,
        echo=not driver.quiet
    )
    rc.datastore.tables.saverow.assert_any_call('disc_learning.csv', [
        'gen_step', 'disc_step', 'Dloss', 'Daccuracy',
        'SSsolve_time', 'gradient_time',
    ])
    discriminator = rc.gan.discriminator
    header = ['gen_step', 'disc_step'] + [
        '{}.nnorm'.format(p.name)  # Normalized NORM
        for p in lasagne.layers.get_all_params(discriminator, trainable=True)
    ]
    rc.datastore.tables.saverow.assert_any_call('disc_param_stats.csv', header)
