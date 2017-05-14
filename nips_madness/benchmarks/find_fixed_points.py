import timeit

from ..ssnode import sample_fixed_points


def make_bench_find_fixed_points(**kwargs):
    def stmt():
        return sample_fixed_points(**kwargs)
    stmt()
    return stmt


def run_benchmarks(repeat=3, **kwargs):
    for io_type in ['asym_power', 'asym_linear', 'asym_tanh']:
        for method in ['serial', 'parallel']:
            stmt = make_bench_find_fixed_points(io_type=io_type,
                                                method=method)
            times = timeit.repeat(stmt, repeat=repeat, number=1)

            print('{: <15} {: <15} min = {:<10.4g}  avg = {:<10.4g}'.format(
                io_type, method, min(times), sum(times) / repeat))


if __name__ == '__main__':
    run_benchmarks()
