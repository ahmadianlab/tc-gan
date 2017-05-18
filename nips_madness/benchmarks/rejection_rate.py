from __future__ import print_function, division

from .. import utils
from ..ssnode import sample_fixed_points
from ..tests import test_dynamics  # ..to emit the warnings here
del test_dynamics


def run_benchmarks(NZ, max_iter_list, **kwargs):
    print()
    print('{:<10} {:<7} {:<7} {:<10} {}'.format(
        'dt', '#rej', '%rej', 'time', 'codes'))
    dt_0 = 0.001
    max_iter_0 = 10000
    for max_iter in max_iter_list:
        dt = dt_0 * max_iter_0 / max_iter
        with utils.StopWatch() as time:
            _zs, _xs, info = sample_fixed_points(
                NZ=NZ, dt=dt, max_iter=max_iter, **kwargs)

        print('{:<10.4g} {:<7} {:<7.2%} {:<10.4g} {}'.format(
            dt,
            info.rejections,
            info.rejections / (info.rejections + NZ),
            time.sum(),
            dict(info.counter)))


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N', default=201, type=int)
    parser.add_argument('--NZ', default=15, type=int)
    parser.add_argument('--max-iter-list',
                        default=[10000, 12500, 20000, 100000],
                        type=lambda x: list(map(int, x.split(','))),
                        help='dt is computed from this.')
    ns = parser.parse_args(args)
    run_benchmarks(**vars(ns))


if __name__ == '__main__':
    main()
