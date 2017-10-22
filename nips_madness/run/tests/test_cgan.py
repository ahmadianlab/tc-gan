import pytest

from .. import cgan
from . import test_gan
from .test_gan import load_json


def single_g_step(args):
    cgan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
        '--WGAN_n_critic0', '1',
    ] + args)


@pytest.mark.parametrize('args', [
    [],
    ['--disc-param-save-interval', '1'],
    ['--disc-param-save-on-error'],
    # These options do not work at the moment:
    pytest.mark.skip(['--track_offset_identity']),
    pytest.mark.skip(['--sample-sites', '0, 0.5']),
])
def test_smoke_slowtest(args, cleancwd):
    single_g_step(args)
    assert cleancwd.join('logfiles').check()

    info = load_json(cleancwd, 'info.json')
    assert info['extra_info']['script_file'] == cgan.__file__


def test_disc_param_save_slowtest(cleancwd):
    test_gan.test_disc_param_save_slowtest(cleancwd, single_g_step)
