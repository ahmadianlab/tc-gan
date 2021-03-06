"""
Benchmark to find a good `resubmit_threshold` for `.find_fixed_points`.
"""

import itertools
import timeit

from .find_fixed_points import make_bench_find_fixed_points


def run_benchmarks(repeat, resubmit_thresholds, io_types, deterministics,
                   NZs, dts, **kwargs):
    for NZ, dt, io_type, resubmit_threshold, deterministic in \
            itertools.product(NZs, dts, io_types, resubmit_thresholds,
                              deterministics):
        stmt = make_bench_find_fixed_points(
            io_type=io_type,
            resubmit_threshold=resubmit_threshold,
            dt=dt,
            NZ=NZ,
            **kwargs)
        times = timeit.repeat(stmt, repeat=repeat, number=1)

        print('{:<3} {:<10.4g} {: <15} {:<9.3g} {} '
              ' min = {:<10.4g} avg = {:<10.4g}'
              .format(NZ, dt, io_type, resubmit_threshold,
                      'D' if deterministic else 'S',
                      min(times), sum(times) / repeat))


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', default=3, type=int)
    parser.add_argument('--N', default=201, type=int)
    parser.add_argument('--resubmit-thresholds',
                        default=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                        type=eval)
    parser.add_argument('--deterministics',
                        default=[True, False],
                        type=eval)
    parser.add_argument('--io-types',
                        default=['asym_power', 'asym_tanh'],
                        type=lambda x: x.split(','))
    parser.add_argument('--dts',
                        default=[0.0008, 0.0005],
                        type=eval)
    parser.add_argument('--NZs',
                        default=[15, 75],
                        type=int)
    ns = parser.parse_args(args)
    run_benchmarks(**vars(ns))


if __name__ == '__main__':
    main()
