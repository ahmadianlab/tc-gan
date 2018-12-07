import os

import pytest


@pytest.fixture()
def cleancwd(tmpdir, request):
    origdir = os.getcwd()
    tmpdir.chdir()

    @request.addfinalizer
    def goback():
        os.chdir(origdir)

    return tmpdir


old_gan = pytest.mark.skipif(
    os.environ.get('TEST_OLD_GAN') != 'yes',
    reason='$TEST_OLD_GAN != yes')
