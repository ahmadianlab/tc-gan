"""
Benchmark forward pass of SSN implemented in Theano.
"""

import json
import itertools
import collections
from logging import getLogger

import numpy as np

from ..networks.fixed_time_sampler import FixedTimeTuningCurveSampler
from ..utils import StopWatch, log_timing

logger = getLogger(__name__)


def params_transposed(params):
    if isinstance(params, collections.OrderedDict):
        return zip(*params.items())
    if isinstance(params, dict):
        keys = sorted(params)
        return keys, [params[k] for k in keys]
    return zip(*params)


def params_product(params):
    keys, values = params_transposed(params)
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def ssn_forward(params, repeat, method):
    from shutil import get_terminal_size
    import pandas

    keys, values = params_transposed(params)
    combinations = list(itertools.product(*values))
    configs = [dict(zip(keys, comb)) for comb in combinations]

    watches = collections.defaultdict(StopWatch)
    sampler_store = {}
    for _ in range(repeat):
        for i, sampler_config in enumerate(configs):
            logger.info('sampler_config = %r', sampler_config)
            try:
                sampler = sampler_store[i]
            except KeyError:
                sampler = FixedTimeTuningCurveSampler \
                    .from_dict(sampler_config)
                with log_timing('sampler.prepare()'):
                    sampler.prepare()
                sampler_store[i] = sampler
            fun = getattr(sampler, method)
            opname = '{}.{}()'.format(type(sampler).__name__, method)
            with log_timing(opname), watches[i]:
                fun()

    df = pandas.DataFrame(combinations, columns=keys)
    for i in range(len(df)):
        for stat in ['mean', 'median', 'min', 'max', 'std']:
            df.loc[i, stat] = getattr(np, stat)(watches[i].times[1:])
        df.loc[i, 'first'] = watches[i].times[0]

    # Manually detect terminal size, since passing "'display.width',
    # None" does not detect terminal size (as advertised in
    # https://pandas.pydata.org/pandas-docs/stable/options.html):
    width, _ = get_terminal_size()

    print()
    print('Results:')
    with pandas.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', width):
        print(df)


def main(args=None):
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)
    parser.add_argument(
        '--method', default='forward',
        choices=('forward', 'compute_trajectories'),
        help="Method to be benchmarked.")
    parser.add_argument(
        '--repeat', type=int, default=4,
        help="Number of samples per combination of parameter.")
    parser.add_argument(
        'params', metavar='JSON',
        default=dict(ssn_impl=['default', 'mapclone']),
        type=json.loads, nargs='?',
        help="JSON representation of key-value pairs.")
    ns = parser.parse_args(args)
    ssn_forward(**vars(ns))


if __name__ == '__main__':
    main()
