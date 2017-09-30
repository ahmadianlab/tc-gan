from __future__ import print_function

import multiprocessing
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_meta_info(packages=[]):
    return dict(
        repository=dict(
            revision=git_output(['git', 'rev-parse', 'HEAD']).rstrip(),
            is_clean=git_is_clean(),
        ),
        python=sys.executable,
        packages={p.__name__: p.__version__ for p in packages},
        argv=sys.argv,
    )


def git_is_clean():
    return git_output(['git', 'status', '--short',
                       '--untracked-files=no']).strip() == ''


def git_output(args):
    return subprocess.check_output(
        args,
        cwd=PROJECT_ROOT,
        universal_newlines=True)


def make_progressbar(quiet=False, **kwds):
    if quiet:
        return lambda xs: xs
    else:
        try:
            import progressbar
        except ImportError:
            def dummy_bar(xs):
                for x in xs:
                    print('.', end='')
                    yield x
                print()
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
        return self

    def __exit__(self, type, value, traceback):
        self.times.append(time.time() - self.pre)

    def sum(self):
        return sum(self.times)


def csv_line(value_parser):
    """
    Return a function that parses a line of comma separated values.

    Examples:

    >>> csv_line(int)('1, 2, 3')
    [1, 2, 3]
    >>> csv_line(float)('0.5, 1.5')
    [0.5, 1.5]

    For example, it can be passed to type argument of
    `argparse.ArgumentParser.add_argument` as follows::

        parser.add_argument(
            ...,
            type=csv_line(float),
            help='Comma separated value of floats')

    """
    def convert(string):
        return list(map(value_parser, string.split(',')))
    return convert


def cpu_count(_environ=os.environ):
    """ Return available number of CPUs; Slurm/PBS-aware version. """
    try:
        return int(_environ['SLURM_CPUS_ON_NODE'])
    except (KeyError, ValueError):
        pass
    try:
        return int(_environ['PBS_NUM_PPN'])
    except (KeyError, ValueError):
        pass
    return multiprocessing.cpu_count()
