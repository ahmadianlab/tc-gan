import glob


def assert_logfile_exists(directory, name):
    paths = list(glob.glob(str(directory.join('logfiles', '*', name))))
    assert len(paths) == 1
