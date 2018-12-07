import pytest

from . import test_gan
from .. import cgan
from ...conftest import old_gan
from .test_gan import load_json


def single_g_step(args):
    cgan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--n_bandwidths', '1',
        '--contrast', '20',
        '--WGAN_n_critic0', '1',
    ] + args)


@old_gan
def test_single_g_step_logfiles_slowtest(cleancwd):
    single_g_step([])
    assert cleancwd.join('logfiles').check(dir=1)


@old_gan
@pytest.mark.parametrize('args', [
    [],
    ['--disc-param-save-interval', '1'],
    ['--disc-param-save-on-error'],
    # These options do not work at the moment:
    pytest.mark.skip(['--track_offset_identity']),
    pytest.mark.skip(['--sample-sites', '0, 0.5']),
])
def test_single_g_step_slowtest(args, cleancwd):
    datastore_name = 'results'
    single_g_step(args + ['--datastore', datastore_name])
    datastore_path = cleancwd.join(datastore_name)
    assert datastore_path.check()

    info = load_json(datastore_path, 'info.json')
    assert info['extra_info']['script_file'] == cgan.__file__
    assert 'PATH' in info['meta_info']['environ']


@old_gan
def test_disc_param_save_slowtest(cleancwd):
    test_gan.test_disc_param_save_slowtest(cleancwd, single_g_step)
