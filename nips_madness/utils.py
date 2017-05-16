from __future__ import print_function

import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_meta_info(packages=[]):
    return dict(
        repository=dict(
            revision=git_output(['git', 'rev-parse', 'HEAD']).rstrip(),
            is_clean=git_output(['git', 'status', '--short',
                                 '--untracked-files=no']).strip() == '',
        ),
        python=sys.executable,
        packages={p.__name__: p.__version__ for p in packages},
    )


def git_output(args):
    return subprocess.check_output(
        args,
        cwd=PROJECT_ROOT,
        universal_newlines=True)


def make_progressbar(quiet=False, **kwds):
    def dummy_bar(x):
        return x

    if quiet:
        return dummy_bar
    else:
        try:
            import progressbar
        except ImportError:
            return dummy_bar
        else:
            return progressbar.ProgressBar(**kwds)


class StopWatch(object):

    """
    Context manager based stop watch:

    >>> sw = StopWatch()
    >>> for _ in range(10):
    ...     with sw:
    ...         time.sleep(0.1)
    >>> print('{:.1f}'.format(sw.sum()))
    1.0

    """

    def __init__(self):
        self.times = []

    def __enter__(self):
        self.pre = time.time()

    def __exit__(self, type, value, traceback):
        self.times.append(time.time() - self.pre)

    def sum(self):
        return sum(self.times)
