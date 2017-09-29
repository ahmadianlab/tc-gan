import pytest

from .. import gan


def single_g_step(args):
    gan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
    ] + args)


@pytest.mark.parametrize('args', [
    [],
    ['--loss', 'CE'],
    ['--track_offset_identity'],
    ['--sample-sites', '0, 0.5'],
])
def test_smoke_slowtest(args, cleancwd):
    single_g_step(args)
    assert cleancwd.join('logfiles').check()


def test_disc_param_save_slowtest(cleancwd, single_g_step=single_g_step):
    single_g_step([
        '--disc-param-save-interval', '1',
        '--datastore', '.',
    ])
    assert cleancwd.join('disc_param', '0.npz').check()
