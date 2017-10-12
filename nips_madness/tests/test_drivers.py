from types import SimpleNamespace
from unittest import mock
import collections
import io

import lasagne
import numpy as np
import pytest

from .. import lasagne_param_file
from .. import drivers
from .. import execution
from ..execution import DataTables
from ..recorders import LearningRecorder
from ..run.gan import init_driver
from ..ssnode import DEFAULT_PARAMS


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

    # Setup fake DataStore:
    datastore = mock.Mock()
    datastore.path.side_effect = dspath = FakeDataStorePath()

    # Setup fake DataTables:
    # Connect mocked saverow to true saverow for testing file contents.
    tables = DataTables(None)   # `directory` ignored since `_open` is patched
    tables._open = dspath
    datastore.tables.saverow.side_effect = tables.saverow

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

    for name in 'JDS':
        fake_shared = mock.Mock()
        fake_shared.get_value.side_effect \
            = lambda: np.arange(4).reshape((2, 2))
        setattr(rc.gan, name, fake_shared)

    # For debugging:
    rc._tables = tables
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
    rc.datastore.tables.saverow.assert_any_call(
        LearningRecorder.filename,
        LearningRecorder.column_names,
        echo=not driver.quiet
    )
    disc_learning_column_names = ['gen_step', 'disc_step']
    disc_learning_column_names.extend(DiscFakeUpdateResults.fields)
    disc_learning_column_names = tuple(disc_learning_column_names)
    rc.datastore.tables.saverow.assert_any_call('disc_learning.csv',
                                                disc_learning_column_names,
                                                echo=False)
    discriminator = rc.gan.discriminator
    disc_param_stats_column_names = ['gen_step', 'disc_step'] + [
        '{}.nnorm'.format(p.name)  # Normalized NORM
        for p in lasagne.layers.get_all_params(discriminator, trainable=True)
    ]
    disc_param_stats_column_names = tuple(disc_param_stats_column_names)
    rc.datastore.tables.saverow.assert_any_call('disc_param_stats.csv',
                                                disc_param_stats_column_names,
                                                echo=False)

    for name, skiprows, width, length in [
            ('learning.csv', 1, len(LearningRecorder.column_names),
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
        return lasagne_param_file.load(stream)

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
