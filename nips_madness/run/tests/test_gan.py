import glob
import json

import pytest

from .. import gan
from ... import recorders
from ...analyzers import load_logfile
from ...conftest import old_gan


def single_g_step(args):
    gan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
        '--n_bandwidths', '1',
        '--WGAN_n_critic0', '1',
    ] + args)


def logfile(directory, name):
    path, = glob.glob(str(directory.join('logfiles', '*', name)))
    return path


def load_table(directory, name):
    return load_logfile(logfile(directory, name))


def load_json(directory, name):
    with open(logfile(directory, name)) as file:
        return json.load(file)


def load_gandata(directory):
    from ...analyzers import load_gandata
    return load_gandata(logfile(directory, 'info.json'))


def make_gan(n_samples, bandwidths, contrast, **run_config):
    return gan.GenerativeAdversarialNetwork(
        NZ=n_samples,
        NB=len(bandwidths) * len(contrast),
        **run_config)


@old_gan
@pytest.mark.parametrize('args', [
    [],
    ['--loss', 'CE'],
    ['--include-inhibitory-neurons'],
    ['--track_offset_identity'],
    ['--track_offset_identity', '--include-inhibitory-neurons'],
    ['--sample-sites', '0, 0.5'],
    ['--disc-param-save-on-error'],
    ['--gen-update', 'rmsprop'],
    ['--gen-param-type', 'clip'],
])
def test_single_g_step_slowtest(args, cleancwd):
    single_g_step(args)
    assert cleancwd.join('logfiles').check()

    info = load_json(cleancwd, 'info.json')
    assert info['extra_info']['script_file'] == gan.__file__
    assert 'PATH' in info['meta_info']['environ']

    ganet = make_gan(**info['run_config'])
    _, n_tc_points = ganet.disc_input_shape

    _, tc_mean = load_table(cleancwd, 'TC_mean.csv')
    assert tc_mean.shape[1] == n_tc_points * 2  # fake and ture

    disc_learning = load_table(cleancwd, 'disc_learning.csv')
    assert len(disc_learning.names) == disc_learning.data.shape[1] == 8

    with pytest.warns(None) as record:
        data = load_gandata(cleancwd)
    assert len(record) == 0

    assert data.main.shape == (1, len(recorders.LearningRecorder.column_names))
    assert data.main_names == list(recorders.LearningRecorder.column_names)

    assert data.gen.shape == (1, len(recorders.GenParamRecorder.column_names))
    assert data.gen_names == list(recorders.GenParamRecorder.column_names)


@old_gan
def test_disc_param_save_slowtest(cleancwd, single_g_step=single_g_step):
    single_g_step([
        '--disc-param-save-interval', '1',
        '--datastore', '.',
    ])
    assert cleancwd.join('disc_param', 'last.npz').check()
