import json

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


@pytest.mark.parametrize('args, config', [
    ([], dict(ssn_type='heteroin')),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin')),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin',
                                            V=[0.3, 0])),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin',
                                            V_min=[0, 0],
                                            V_max=[1, 0])),
])
def test_single_g_step_with_load_config_slowtest(args, config,
                                                 cleancwd, **kwargs):
    config = dict(config)
    if config.get('ssn_type') == 'heteroin':
        config.setdefault('dataset_provider', 'fixedtime')
        # ...since, at the moment, dataset_provider='ssnode' does not
        # work with ssn_type='heteroin'.

    config_path = str(cleancwd.join('run.json'))
    with open(config_path, 'w') as file:
        json.dump(config, file)

    args += ['--load-config', config_path]
    test_single_g_step_slowtest(args, cleancwd, **kwargs)
