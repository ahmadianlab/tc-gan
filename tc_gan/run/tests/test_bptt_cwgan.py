import pytest

from . import test_bptt_wgan
from .. import bptt_cwgan


def single_g_step(args):
    bptt_cwgan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--num-models', '2',
        '--n_bandwidths', '1',
        '--WGAN_n_critic0', '1',
        '--seqlen', '4',
        '--skip-steps', '2',
        '--tc-stats-record-interval', '1',
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

    assert cleancwd.join('results', 'store.hdf5').check()


@pytest.mark.parametrize('args, config', [
    ([], dict(ssn_type='heteroin')),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin')),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin',
                                            V=[0.3, 0])),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin',
                                            gen_V_min=[0, 0],
                                            gen_V_max=[1, 0])),
    ([], dict(ssn_type='deg-heteroin')),
    (['--include-inhibitory-neurons'], dict(ssn_type='deg-heteroin', V=0.5)),
])
def test_single_g_step_with_load_config_slowtest(args, config, cleancwd):
    test_bptt_wgan.test_single_g_step_with_load_config_slowtest(
        args, config, cleancwd,
        single_g_step=single_g_step,
        script_file=bptt_cwgan.__file__)
