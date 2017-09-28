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
