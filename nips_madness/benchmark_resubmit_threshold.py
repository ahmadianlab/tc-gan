import timeit

from .benchmark_find_fixed_points import make_bench_find_fixed_points


def run_benchmarks(repeat, resubmit_thresholds, io_types, **kwargs):
    for io_type in io_types:
        for resubmit_threshold in resubmit_thresholds:
            stmt = make_bench_find_fixed_points(
                io_type=io_type,
                resubmit_threshold=resubmit_threshold,
                **kwargs)
            times = timeit.repeat(stmt, repeat=repeat, number=1)

            print('{:<10.4g} {: <15} min = {:<10.4g}  avg = {:<10.4g}'.format(
                resubmit_threshold, io_type,
                min(times), sum(times) / repeat))


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', default=3, type=int)
    parser.add_argument('--resubmit-thresholds',
                        default=[0, 0.05] + [i * 0.1 for i in range(1, 10)],
                        type=eval)
    parser.add_argument('--io-types',
                        default=['asym_power', 'asym_tanh'],
                        type=lambda x: x.split(','))
    ns = parser.parse_args(args)
    run_benchmarks(**vars(ns))


if __name__ == '__main__':
    main()
