#!/usr/bin/env python

"""
Convenient entry point to run modules.

Examples::

  ./run nips_madness/benchmarks/solve_dynamics.py
  ./run nips_madness/benchmarks/resubmit_threshold.py -- --repeat=10

"""

from __future__ import print_function

import importlib
import logging
import os
import pdb
import subprocess
import sys
import traceback

record_env_script_template = '''
module list &> module-list.txt || rm module-list.txt
set -ex
which python
which pip
which conda
pip freeze > pip-freeze.txt
conda list --prefix "{project_root}/env" --export > conda-list.txt
'''


def run_module(module, arguments, use_pdb, use_pudb,
               assert_repo_is_clean,
               record_env, mpl_style,
               log_level, log_format, log_datefmt):
    logging.basicConfig(level=getattr(logging, log_level),
                        format=log_format, datefmt=log_datefmt)
    here = os.path.realpath(os.path.dirname(__file__))
    if os.path.isfile(module) and module.endswith('.py'):
        relpath = os.path.relpath(os.path.realpath(module), here)
        module = relpath[:-len('.py')].replace(os.path.sep, '.')
    from nips_madness.execution import KnownError
    loaded = importlib.import_module(module)
    if not hasattr(loaded, 'main'):
        print('Module', module, 'do not have main function.')
        return 1
    if assert_repo_is_clean:
        from nips_madness.utils import git_is_clean
        if not git_is_clean():
            print('Repository is not clean.')
            return 3
    if record_env:
        subprocess.check_call(
            record_env_script_template.format(project_root=here),
            shell=True,
            executable='/bin/bash',
            cwd=record_env)
    if mpl_style:
        import matplotlib
        matplotlib.style.use(mpl_style)
    try:
        loaded.main(arguments)
    except Exception as err:
        if use_pdb:
            traceback.print_exc()
            print()
            pdb.post_mortem()
        elif use_pudb:
            import pudb
            pudb.post_mortem()
        elif isinstance(err, KnownError):
            print(err)
            return err.exit_code
        else:
            raise


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=type('FormatterClass',
                             (argparse.RawDescriptionHelpFormatter,
                              argparse.ArgumentDefaultsHelpFormatter),
                             {}),
        description=__doc__)
    parser.add_argument(
        'module',
        help='''
        File path (e.g., nips_madness/benchmarks/solve_dynamics.py) to
        a module or "dotted import path" (e.g.,
        nips_madness.benchmarks.resubmit_threshold).
        ''')
    parser.add_argument(
        'arguments', nargs='*',
        help="arguments passed to module's main function")
    parser.add_argument(
        '--log-level', default='INFO',
        choices='CRITICAL ERROR WARNING INFO DEBUG NOTSET'.split(),
        help='''Logging level.
        See: https://docs.python.org/3/library/logging.html#levels''')
    parser.add_argument(
        '--log-format',
        default='%(asctime)s %(levelname)s %(message)s',
        help='''Logging format. For example, to include logger name,
        use '%(asctime)s %(levelname)s %(name)s %(message)s'.
        See:
        https://docs.python.org/3/library/logging.html#logrecord-attributes''')
    parser.add_argument(
        '--log-datefmt',
        default='%Y-%m-%d %H:%M:%S',
        help='''Logging datetime format.
        See: https://docs.python.org/3/library/time.html#time.strftime''')
    parser.add_argument(
        '--pdb', action='store_true', dest='use_pdb',
        help='drop into pdb when there is an exception')
    parser.add_argument(
        '--pudb', action='store_true', dest='use_pudb',
        help='drop into pupdb when there is an exception')
    parser.add_argument(
        '--assert-repo-is-clean', action='store_true',
        help='abort (with code 3) if this repository is not clean')
    parser.add_argument(
        '--record-env',
        help='''Directory in which environment information is saved.
        Do nothing if not given.''')
    parser.add_argument(
        '--mpl-style',
        help='If given, call matplotlib.style.use.')
    ns = parser.parse_args(args)
    sys.exit(run_module(**vars(ns)))


if __name__ == '__main__':
    main()
