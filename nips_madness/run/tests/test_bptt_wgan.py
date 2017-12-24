import json

import pytest

from .. import bptt_wgan
from ... import recorders
from ...loaders import load_records
from .test_gan import load_json


def single_g_step(args):
    bptt_wgan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--n_bandwidths', '1',
        '--WGAN_n_critic0', '1',
        '--seqlen', '4',
        '--skip-steps', '2',
    ] + args)


def test_single_g_step_logfiles_slowtest(cleancwd):
    single_g_step([])
    assert cleancwd.join('logfiles').check(dir=1)


@pytest.mark.parametrize('args', [
    [],
    ['--sample-sites', '0, 0.5'],
    ['--include-inhibitory-neurons'],
])
def test_single_g_step_slowtest(args, cleancwd,
                                single_g_step=single_g_step,
                                script_file=bptt_wgan.__file__):
    datastore_name = 'results'
    single_g_step(args + ['--datastore', datastore_name])
    datastore_path = cleancwd.join(datastore_name)
    assert datastore_path.check()

    info = load_json(datastore_path, 'info.json')
    assert info['extra_info']['script_file'] == script_file
    assert 'PATH' in info['meta_info']['environ']

    if info['run_config'].get('ssn_type') == 'heteroin':
        # Skip load_gandata test, since it's not implemented yet.  # FIXME
        return

    with pytest.warns(None) as record:
        rec = load_records(str(datastore_path))
    assert len(record) == 0

    learning_df = rec.learning
    desired = list(recorders.LearningRecorder.dtype.names) + ['epoch']
    assert list(learning_df.columns) == desired
    assert len(learning_df) == 1

    generator_df = rec.generator
    names = list(recorders.GenParamRecorder.dtype.names) + ['epoch']
    assert list(generator_df.columns) == names
    assert len(generator_df) == 1


@pytest.mark.parametrize('args, config', [
    ([], dict(ssn_type='heteroin')),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin')),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin',
                                            V=[0.3, 0])),
    (['--include-inhibitory-neurons'], dict(ssn_type='heteroin',
                                            gen_V_min=[0, 0],
                                            gen_V_max=[1, 0])),
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
