import pytest

from .. import bptt_moments
from .test_gan import load_json


def single_g_step(args):
    bptt_moments.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--n_bandwidths', '1',
        '--seqlen', '4',
        '--skip-steps', '2',
    ] + args)


@pytest.mark.parametrize('args', [
    [],
    ['--sample-sites', '0, 0.5'],
    ['--include-inhibitory-neurons'],
])
def test_single_g_step_slowtest(args, cleancwd):
    datastore_name = 'results'
    single_g_step(args + ['--datastore', datastore_name])
    datastore_path = cleancwd.join(datastore_name)
    assert datastore_path.check()

    info = load_json(datastore_path, 'info.json')
    assert info['extra_info']['script_file'] == bptt_moments.__file__
    assert 'PATH' in info['meta_info']['environ']
