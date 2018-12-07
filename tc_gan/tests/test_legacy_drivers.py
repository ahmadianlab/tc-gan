from __future__ import print_function

from types import SimpleNamespace
from unittest import mock
import collections
import functools
import io

import lasagne
import numpy as np
import pytest

from .. import drivers
from .. import execution
from .. import recorders
from ..execution import DataTables, DataStore
from ..lasagne_toppings import param_file
from ..run.gan import init_driver
from ..ssnode import DEFAULT_PARAMS


class CSVLearningRecorder(recorders.CSVRecorder,
                          recorders.LegacyLearningRecorder):
    pass


class CSVGenParamRecorder(recorders.CSVRecorder,
                          recorders.GenParamRecorder):
    pass


class CSVDiscParamStatsRecorder(recorders.CSVRecorder,
                                recorders.DiscParamStatsRecorder):
    pass


class CSVDiscLearningRecorder(recorders.CSVRecorder,
                              recorders.DiscLearningRecorder):
    pass


class LegacyGANDriverWithCSV(drivers.GANDriver):

    def make_learning_recorder(self):
        return CSVLearningRecorder.from_driver(self)

    def make_generator_recorder(self):
        return CSVGenParamRecorder.from_driver(self)

    def make_discparamstats_recorder(self):
        return CSVDiscParamStatsRecorder.from_driver(self)

    def make_disclearning_recorder(self):
        return CSVDiscLearningRecorder.from_driver(self)


class FakeDataStorePath(object):

    def __init__(self):
        self.returned = collections.defaultdict(list)

    def __call__(self, *args):
        if args[-1].endswith('.csv'):
            stream = io.StringIO()
        else:
            stream = io.BytesIO()
        self.returned[args].append(stream)
        return stream


def make_discriminator():
    l0 = lasagne.layers.InputLayer((2, 3))
    l1 = lasagne.layers.DenseLayer(l0, 5)
    return l1


def fake_datastore():
    # Setup fake DataStore:
    datastore = mock.Mock()
    datastore.path.side_effect = dspath = FakeDataStorePath()

    # Setup fake DataTables:
    # Connect mocked saverow to true saverow for testing file contents.
    tables = DataTables(None)   # `directory` ignored since `_open` is patched
    tables._open = dspath
    datastore.tables.saverow.side_effect = tables.saverow

    datastore.save_exit_reason = functools.partial(DataStore.save_exit_reason,
                                                   datastore)

    # For debugging:
    datastore._tables = tables
    return datastore


def setup_fake_gan(gan):
    gan.loss_type = 'WD'
    gan.discriminator = make_discriminator()
    gan.NZ = 3  # (n_samples)

    for i, name in enumerate('JDS'):
        fake_shared = mock.Mock()
        fake_shared.get_value.return_value \
            = np.arange(4).reshape((2, 2)) + 10 ** i
        setattr(gan, name, fake_shared)

    gan.get_gen_param = lambda: (gan.J.get_value(),
                                 gan.D.get_value(),
                                 gan.S.get_value())


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

    rc = SimpleNamespace(**init_driver(
        fake_datastore(),
        iterations=iterations,
        quiet=quiet,
        disc_param_save_interval=disc_param_save_interval,
        disc_param_template=disc_param_template,
        disc_param_save_on_error=disc_param_save_on_error,
        quit_JDS_threshold=quit_JDS_threshold,
        J0=J0,
        D0=D0,
        S0=S0,
        driver_class=LegacyGANDriverWithCSV,
        **run_config))

    setup_fake_gan(rc.gan)

    def patched_pre_loop():
        real_pre_loop()
        # "Turn off" SSNRejectionLimiter:
        rc.driver.rejection_limiter.rejection_limit = 1
    real_pre_loop = rc.driver.pre_loop
    rc.driver.pre_loop = patched_pre_loop

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
    fields = ('Dloss', 'Daccuracy', 'SSsolve_time', 'gradient_time',
              'model_convergence', 'model_unused')
    size = len(fields)

    def make_result(self):
        result = super(DiscFakeUpdateResults, self).make_result()
        model_info = SimpleNamespace(
            rejections=result[-2],
            unused=result[-1],
        )
        return list(result[:-2]) + [model_info]


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
        'rate_penalty',
        'dynamics_penalty',
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

    empty_calls = [mock.call()] * iterations
    assert rc.gan.J.get_value.call_args_list == empty_calls
    assert rc.gan.D.get_value.call_args_list == empty_calls
    assert rc.gan.S.get_value.call_args_list == empty_calls

    rc.datastore.path.assert_called_with('disc_param', 'last.npz')

    # Make sure headers (column names) are saved:
    learning_recorder = rc.driver.learning_recorder
    rc.datastore.tables.saverow.assert_any_call(
        learning_recorder.filename,
        learning_recorder.column_names,
        echo=not driver.quiet
    )
    disc_learning_column_names = ['gen_step', 'disc_step']
    disc_learning_column_names.extend(DiscFakeUpdateResults.fields)
    disc_learning_column_names = tuple(disc_learning_column_names)
    rc.datastore.tables.saverow.assert_any_call('disc_learning.csv',
                                                disc_learning_column_names,
                                                echo=False)

    discriminator = rc.gan.discriminator
    disc_param_stats_column_names = ['gen_step', 'disc_step']
    disc_param_stats_column_names.extend(
        rc.driver.discparamstats_recorder.disc_param_unique_names(
            p.name for p in
            lasagne.layers.get_all_params(discriminator, trainable=True)))
    disc_param_stats_column_names = tuple(disc_param_stats_column_names)
    rc.datastore.tables.saverow.assert_any_call('disc_param_stats.csv',
                                                disc_param_stats_column_names,
                                                echo=False)

    for name, skiprows, width, length in [
            ('learning.csv', 1, len(learning_recorder.column_names),
             iterations),
            ('disc_learning.csv', 1, len(disc_learning_column_names),
             iterations * n_critic),
            ('disc_param_stats.csv', 1, len(disc_param_stats_column_names),
             iterations * n_critic),
            ('generator.csv', 1, 13, iterations),
            ]:
        # "Assert" that DataTables._open is called once:
        stream, = rc.datastore.path.side_effect.returned[name, ]
        stream.seek(0)
        loaded = np.loadtxt(stream, delimiter=',', skiprows=skiprows)
        if loaded.ndim == 1:
            loaded = loaded.reshape((1, -1))
        assert loaded.shape == (length, width)

    def load_json(name):
        for (obj, filename), _ in rc.datastore.dump_json.call_args_list:
            if filename == name:
                return obj

    exit_info = load_json('exit.json')
    assert exit_info == dict(
        reason='end_of_iteration',
        good=True,
    )

    def lasagne_load(name):
        stream, = rc.datastore.path.side_effect.returned['disc_param', name]
        stream.seek(0)
        return param_file.load(stream)

    stored_values = lasagne_load(driver.disc_param_template)
    desired_values = lasagne.layers.get_all_param_values(rc.gan.discriminator)
    np.testing.assert_equal(stored_values, desired_values)


def test_disc_param_isfinite():
    datastore = mock.Mock()
    l0 = lasagne.layers.InputLayer((2, 3))
    l1 = lasagne.layers.DenseLayer(l0, 5)
    W = l1.W.get_value()
    W[0, 0] = np.nan
    l1.W.set_value(W)
    nnorms = np.array([np.nan])
    with pytest.raises(execution.KnownError):
        drivers.check_disc_param(datastore, l1, nnorms)


def test_quit_JDS_threshold_quit():
    datastore = mock.Mock()
    JDS_true = [DEFAULT_PARAMS[k] for k in 'JDS']
    log_JDS = [  # from t017/28
        [[-2.725132882048388, -2.4531698490543286],
         [-2.1680251198864506, -3.1575330875403287]],
        [[-0.6751161156746839, -0.3826601625506246],
         [-0.34232003427022, -1.1335422836893538]],
        [[-2.7327504049935936, -4.210179719643937],
         [-1.9547447679855652, -3.5791486928972325]],
    ]
    with pytest.raises(execution.KnownError):
        drivers.maybe_quit(
            datastore,
            JDS_fake=list(map(np.exp, log_JDS)),
            JDS_true=JDS_true,
            quit_JDS_threshold=0.4,
        )
    datastore.dump_json.assert_called_once()


def test_quit_JDS_threshold_noquit():
    datastore = mock.Mock()
    JDS_true = [DEFAULT_PARAMS[k] for k in 'JDS']
    JDS_fake = np.array(JDS_true) + 0.01
    drivers.maybe_quit(
        datastore,
        JDS_fake=JDS_fake,
        JDS_true=JDS_true,
        quit_JDS_threshold=0.4,
    )
    datastore.dump_json.assert_not_called()


def test_rejection_limiter_should_abort():
    limiter = drivers.SSNRejectionLimiter(None, n_samples=10)
    over_limit = 20
    under_limit = 0

    assert not limiter.should_abort(over_limit)

    limiter.should_abort(under_limit)  # reset
    max_over_limit = [limiter.should_abort(over_limit)
                      for _ in range(limiter.max_consecutive_exceedings)]
    assert not any(max_over_limit)
    assert not limiter.should_abort(under_limit)

    limiter.should_abort(under_limit)  # reset
    max_over_limit = [limiter.should_abort(over_limit)
                      for _ in range(limiter.max_consecutive_exceedings)]
    assert not any(max_over_limit)
    assert limiter.should_abort(over_limit)


def test_rejection_limiter_exception():
    datastore = mock.Mock()
    limiter = drivers.SSNRejectionLimiter(datastore, n_samples=10)
    over_limit = 20

    for _ in range(limiter.max_consecutive_exceedings):
        limiter(over_limit)

    with pytest.raises(execution.KnownError):
        limiter(over_limit)

    datastore.dump_json.assert_called_once_with(dict(
        reason='too_many_rejections',
        good=False,
    ), 'exit.json')


@pytest.mark.parametrize('repeats, shifts, last_shift', [
    ([9], [+1], +1),
    ([9], [+1], -1),
    ([0, 5, 4], [-1, +1, -1], +1),
    ([1, 5, 3], [-1, +1, -1], +1),
    ([2, 5, 2], [-1, +1, -1], +1),
    ([3, 5, 1], [-1, +1, -1], +1),
    ([4, 5, 0], [-1, +1, -1], +1),
])
def test_wgan_dloss_limiter_exception(repeats, shifts, last_shift):
    datastore = mock.Mock()
    limiter = drivers.WGANDiscLossLimiter(datastore,
                                          prob_limit=0.6 - 1e-5,
                                          hist_length=10)

    print()
    for i, (num, shift) in enumerate(zip(repeats, shifts)):
        print('i={} shift={:+}: '.format(i, shift), end='')
        for j in range(num):
            print('+' if shift > 0 else '-', end='')

            limiter(limiter.wild_disc_loss + shift)
        print()

    with pytest.raises(execution.KnownError):
        limiter(limiter.wild_disc_loss + last_shift)

    datastore.dump_json.assert_called_once_with(dict(
        reason='wild_disc_loss',
        good=False,
    ), 'exit.json')
