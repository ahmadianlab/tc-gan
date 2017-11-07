import pytest

from . import test_bptt_wgan
from .. import bptt_cwgan
from .utils import assert_logfile_exists


def single_g_step(args):
    bptt_cwgan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--num-models', '2',
        '--n_bandwidths', '1',
        '--WGAN_n_critic0', '1',
        '--seqlen', '4',
        '--skip-steps', '2',
    ] + args)


@pytest.mark.parametrize('args', [
    [],
    ['--num-models', '1'],
    ['--sample-sites', '0, 0.5'],
    ['--contrasts', '5, 20'],
    ['--include-inhibitory-neurons'],
])
def test_single_g_step_slowtest(args, cleancwd):
    test_bptt_wgan.test_single_g_step_slowtest(
        args, cleancwd,
        single_g_step=single_g_step,
        script_file=bptt_cwgan.__file__)

    assert_logfile_exists(cleancwd, 'store.hdf5')
