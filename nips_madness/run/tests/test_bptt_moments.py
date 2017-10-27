import pytest

from .. import bptt_moments
from .test_gan import load_json


def single_g_step(args):
    bptt_moments.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
    ] + args)


@pytest.mark.parametrize('args', [
    [],
    ['--sample-sites', '0, 0.5'],
    ['--include-inhibitory-neurons'],
])
def test_single_g_step_slowtest(args, cleancwd):
    single_g_step(args)
    assert cleancwd.join('logfiles').check()

    info = load_json(cleancwd, 'info.json')
    assert info['extra_info']['script_file'] == bptt_moments.__file__
