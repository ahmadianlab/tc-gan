import json

import pytest

from .. import bptt_moments
from ... import recorders
from ...loaders import load_records
from ...networks.tests.test_tuning_curve import flat_param_names
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
    ssn_type = info['run_config'].get('ssn_type', 'default')

    with pytest.warns(None) as warned:
        rec = load_records(str(datastore_path))
    assert len(warned) == 0

    learning_df = rec.learning
    desired = list(recorders.MMLearningRecorder.dtype.names) + ['epoch']
    assert list(learning_df.columns) == desired
    assert len(learning_df) == 1

    generator_df = rec.generator
    names = list(recorders.GenParamRecorder.dtype.names) + ['epoch']
    if ssn_type == 'heteroin':
        i = names.index('J_EE')
        assert i >= 0
        names = names[:i] + ['V_E', 'V_I'] + names[i:]
    assert list(generator_df.columns) == names
    assert len(generator_df) == 1

    if ssn_type in ('heteroin', 'deg-heteroin'):
        assert rec.param_array_names == ['V', 'J', 'D', 'S']
    else:
        assert rec.param_array_names == ['J', 'D', 'S']
    assert rec.param_element_names == flat_param_names[ssn_type]


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
