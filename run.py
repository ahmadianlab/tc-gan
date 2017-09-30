#!/usr/bin/env python

"""
Convenient entry point to run modules.

Examples::

  ./run nips_madness/benchmarks/solve_dynamics.py
  ./run nips_madness/benchmarks/resubmit_threshold.py -- --repeat=10

"""

from __future__ import print_function

import importlib
import os
import pdb
import sys
import traceback


def run_module(module, arguments, use_pdb, assert_repo_is_clean):
    if os.path.isfile(module) and module.endswith('.py'):
        here = os.path.realpath(os.path.dirname(__file__))
        relpath = os.path.relpath(os.path.realpath(module), here)
        module = relpath[:-len('.py')].replace(os.path.sep, '.')
    loaded = importlib.import_module(module)
    if not hasattr(loaded, 'main'):
        print('Module', module, 'do not have main function.')
        return 1
    if assert_repo_is_clean:
        from nips_madness.utils import git_is_clean
        if not git_is_clean():
            print('Repository is not clean.')
            return 3
    try:
        loaded.main(arguments)
    except Exception:
        if use_pdb:
            traceback.print_exc()
            print()
            pdb.post_mortem()
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
        '--pdb', action='store_true', dest='use_pdb',
        help='drop into pdb when there is an exception')
    parser.add_argument(
        '--assert-repo-is-clean', action='store_true',
        help='abort (with code 3) if this repository is not clean')
    ns = parser.parse_args(args)
    sys.exit(run_module(**vars(ns)))


if __name__ == '__main__':
    main()
