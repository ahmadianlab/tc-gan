"""
Print wall-time comparing serial/parallel; asym_power/tanh/...; Nz=30,150.
"""

import itertools
import timeit

from ..ssnode import sample_fixed_points


def make_bench_find_fixed_points(**kwargs):
    def stmt():
        return sample_fixed_points(**kwargs)
    return stmt


def run_benchmarks(io_types, methods, NZs, repeat=3, **kwargs):
    for io_type, method, NZ in itertools.product(
            io_types, methods, NZs):

        stmt = make_bench_find_fixed_points(io_type=io_type,
                                            NZ=NZ,
                                            method=method)
        times = timeit.repeat(stmt, repeat=repeat, number=1)

        print('{: <15} {: <10} {: <5} min = {:<10.4g}  avg = {:<10.4g}'.format(
            io_type, method, NZ, min(times), sum(times) / repeat))


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', default=3, type=int)
    parser.add_argument('--methods',
                        default=['serial', 'parallel'],
                        type=lambda x: x.split(','))
    parser.add_argument('--io-types',
                        default=['asym_power', 'asym_tanh'],
                        type=lambda x: x.split(','))
    parser.add_argument('--NZs',
                        default=[30, 150],
                        type=lambda x: list(map(int, x.split(','))))
    ns = parser.parse_args(args)
    run_benchmarks(**vars(ns))


if __name__ == '__main__':
    main()
