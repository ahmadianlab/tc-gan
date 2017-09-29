import pytest

from .. import cgan


@pytest.mark.parametrize('args', [
    [],
    ['--disc-param-save-interval', '1'],
    # These options do not work at the moment:
    pytest.mark.xfail(['--track_offset_identity']),
    pytest.mark.xfail(['--sample-sites', '0, 0.5']),
])
def test_smoke_slowtest(args, cleancwd):
    cgan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
        '--WGAN_n_critic0', '1',
    ] + args)
    assert cleancwd.join('logfiles').check()
