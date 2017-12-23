import json

import h5py
import pytest

from .. import gan
from ... import recorders
from ...conftest import old_gan
from ...loaders import load_records


def single_g_step(args):
    gan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
        '--n_bandwidths', '1',
        '--WGAN_n_critic0', '1',
    ] + args)


def load_table(directory, name):
    path = directory.join('store.hdf5')
    if not path.check():
        path = directory.join(name + '.hdf5')
    with h5py.File(str(path)) as file:
        return file[name]


def load_json(directory, name):
    with open(str(directory.join(name))) as file:
        return json.load(file)


def make_gan(n_samples, bandwidths, contrast, **run_config):
    return gan.GenerativeAdversarialNetwork(
        NZ=n_samples,
        NB=len(bandwidths) * len(contrast),
        **run_config)


@old_gan
def test_single_g_step_logfiles_slowtest(cleancwd):
    single_g_step([])
    assert cleancwd.join('logfiles').check(dir=1)


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
    datastore_name = 'results'
    single_g_step(args + ['--datastore', datastore_name])
    datastore_path = cleancwd.join(datastore_name)
    assert datastore_path.check()

    info = load_json(datastore_path, 'info.json')
    assert info['extra_info']['script_file'] == gan.__file__
    assert 'PATH' in info['meta_info']['environ']

    ganet = make_gan(**info['run_config'])
    _, n_tc_points = ganet.disc_input_shape

    tc_mean = load_table(datastore_path, 'TC_mean')
    assert tc_mean.shape[1] == n_tc_points * 2  # fake and ture

    disc_learning = load_table(datastore_path, 'disc_learning')
    assert disc_learning.dtype == recorders.DiscLearningRecorder.dtype

    with pytest.warns(None) as record:
        rec = load_records(str(datastore_path))
    assert len(record) == 0

    df = rec.learning
    assert df.column.names == len(recorders.LearningRecorder.column_names)
    assert len(df) == 1

    df = rec.generator
    assert df.column.names == len(recorders.GenParamRecorder.column_names)
    assert len(df) == 1


@old_gan
def test_disc_param_save_slowtest(cleancwd, single_g_step=single_g_step):
    single_g_step([
        '--disc-param-save-interval', '1',
        '--datastore', '.',
    ])
    assert cleancwd.join('disc_param', 'last.npz').check()
