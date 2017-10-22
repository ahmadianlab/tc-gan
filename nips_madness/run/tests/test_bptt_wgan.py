import pytest

from .. import bptt_wgan
from ... import recorders
from .test_gan import load_json, load_gandata


def single_g_step(args):
    bptt_wgan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--WGAN_n_critic0', '1',
    ] + args)


@pytest.mark.parametrize('args', [
    [],
    ['--sample-sites', '0, 0.5'],
])
def test_single_g_step_slowtest(args, cleancwd):
    single_g_step(args)
    assert cleancwd.join('logfiles').check()

    info = load_json(cleancwd, 'info.json')
    assert info['extra_info']['script_file'] == bptt_wgan.__file__

    with pytest.warns(None) as record:
        data = load_gandata(cleancwd)
    assert len(record) == 0

    assert data.main.shape == (1, len(recorders.LearningRecorder.column_names))
    assert data.main_names == list(recorders.LearningRecorder.column_names)

    assert data.gen.shape == (1, len(recorders.GenParamRecorder.column_names))
    assert data.gen_names == list(recorders.GenParamRecorder.column_names)
