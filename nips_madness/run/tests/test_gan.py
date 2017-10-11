import glob

import pytest

from .. import gan
from ...analyzers import load_logfile


def single_g_step(args):
    gan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
        '--WGAN_n_critic0', '1',
    ] + args)


def load_table(directory, name):
    path, = glob.glob(str(directory.join('logfiles', '*', name)))
    return load_logfile(path)


@pytest.mark.parametrize('args', [
    [],
    ['--loss', 'CE'],
    ['--track_offset_identity'],
    ['--sample-sites', '0, 0.5'],
    ['--disc-param-save-on-error'],
    ['--gen-update', 'rmsprop'],
])
def test_smoke_slowtest(args, cleancwd):
    single_g_step(args)
    assert cleancwd.join('logfiles').check()

    n_bandwidths = 4
    _, tc_mean = load_table(cleancwd, 'TC_mean.csv')
    assert tc_mean.shape[1] == n_bandwidths * 2

    disc_learning = load_table(cleancwd, 'disc_learning.csv')
    assert len(disc_learning.names) == disc_learning.data.shape[1] == 8


def test_disc_param_save_slowtest(cleancwd, single_g_step=single_g_step):
    single_g_step([
        '--disc-param-save-interval', '1',
        '--datastore', '.',
    ])
    assert cleancwd.join('disc_param', 'last.npz').check()
