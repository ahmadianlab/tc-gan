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
    with h5py.File(str(path), 'r') as file:
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

    with pytest.warns(None) as warned:
        rec = load_records(str(datastore_path))
    assert len(warned) == 1
    assert 'Loading legacy GAN' in warned.list[0].message.args[0]

    learning_df = rec.learning
    desired = list(recorders.LearningRecorder.dtype.names) + ['epoch']
    assert list(learning_df.columns) == desired
    assert len(learning_df) == 1

    generator_df = rec.generator
    names = list(recorders.GenParamRecorder.dtype.names) + ['epoch']
    assert list(generator_df.columns) == names
    assert len(generator_df) == 1

    disc_learning_df = rec.disc_learning
    desired = list(recorders.DiscLearningRecorder.dtype.names) + ['epoch']
    assert list(disc_learning_df.columns) == desired
    assert len(disc_learning_df) == 1

    ganet = make_gan(**info['run_config'])
    _, n_tc_points = ganet.disc_input_shape
    tc_mean_df = rec.TC_mean
    assert list(tc_mean_df['gen'].columns) == list(range(n_tc_points))
    assert list(tc_mean_df['data'].columns) == list(range(n_tc_points))
    tc_mean_df['gen_step']  # should be accessible
    tc_mean_df['epoch']     # should be accessible


@old_gan
def test_disc_param_save_slowtest(cleancwd, single_g_step=single_g_step):
    single_g_step([
        '--disc-param-save-interval', '1',
        '--datastore', '.',
    ])
    assert cleancwd.join('disc_param', 'last.npz').check()
