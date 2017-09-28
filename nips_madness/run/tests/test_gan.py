import pytest

from .. import gan


@pytest.mark.parametrize('args', [
    [],
    ['--loss', 'CE'],
    ['--track_offset_identity'],
    ['--sample-sites', '0, 0.5'],
])
def test_smoke_slowtest(args, cleancwd):
    gan.main([
        '--iterations', '1',
        '--truth_size', '1',
        '--n_samples', '1',
        '--contrast', '20',
    ] + args)
    assert cleancwd.join('logfiles').check()
