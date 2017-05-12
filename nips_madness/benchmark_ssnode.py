import timeit

import numpy as np

import stimuli
from .ssnode import solve_dynamics, solve_dynamics_python
from .tests.test_dynamics import numeric_w


def make_bench_solve_dynamics(
        fun=solve_dynamics,
        param_type='true', seed=0,
        bandwidth=1, smoothness=0.25/8, contrast=20,
        N=51, k=0.01, n=2.2, io_type='asym_tanh', **kwds):
    if param_type == 'true':
        J = np.array([[.0957, .0638], [.1197, .0479]])
        D = np.array([[.7660, .5106], [.9575, .3830]])
        S = np.array([[.6667, .2], [1.333, .2]]) / 8
    elif param_type == 'bad':
        J = np.array([[0.0532592, 0.0396442], [0.0724497, 0.03049]])
        D = np.array([[0.417151, 0.28704], [0.568, 0.236162]])
        S = np.array([[0.0483366, 0.0149695], [0.126188, 0.0149596]])
    else:
        raise ValueError("Unknown param_type: {}".format(param_type))

    if io_type == 'asym_linear':
        # Boost inhibition if io_type=='asym_linear'; otherwise the
        # activity diverges.
        J[:, 1] *= 1.7

    rs = np.random.RandomState(seed)
    Z = rs.rand(1, 2*N, 2*N)
    W, = numeric_w(Z, J, D, S)

    X = np.linspace(-0.5, 0.5, N)
    ext, = stimuli.input([bandwidth], X, smoothness, contrast)
    r0 = np.zeros(2*N)

    kwds = dict(W=W, ext=ext, r0=r0, k=k, n=n, io_type=io_type, **kwds)

    def stmt():
        return fun(**kwds)

    # Call the function once here and make sure it returns a non-NaN FP.
    assert not np.isnan(stmt()).any()

    return stmt


def find_slow_seed(
        repeat=1, top=10,
        # Bad seeds with: io_type='asym_tanh'
        # seeds=[322, 357, 795, 218, 265, 97],
        # Bad seeds with: io_type='asym_linear'
        seeds=[65, 521, 154, 340, 334, 813, 736, 530, 198, 284, 707][:5],
        # seeds=range(1000),
        param_type='true', N=102, **kwds):
    kwds = dict(param_type=param_type, N=N, **kwds)
    data = []
    for s in seeds:
        stmt = make_bench_solve_dynamics(seed=s, **kwds)
        times = timeit.repeat(stmt, repeat=repeat, number=1)
        data.append((min(times), times, s))
    data.sort(reverse=True)
    for min_time, times, s in data[:top]:
        print('seed = {: <5}  min = {:<10.4g}  avg = {:<10.4g}'.format(
            s, min_time, sum(times) / repeat))
    return data


def run_benchmarks(repeat=3):
    for name, target in [
            ('asym_linear (C)',
             make_bench_solve_dynamics()),
            ('asym_linear (Py)',
             make_bench_solve_dynamics(solve_dynamics_python)),
            ('asym_tanh (C)',
             make_bench_solve_dynamics(io_type='asym_tanh')),
            ('asym_tanh (Py)',
             make_bench_solve_dynamics(solve_dynamics_python,
                                       io_type='asym_tanh')),
            ]:
        times = timeit.repeat(target, repeat=repeat, number=1)

        print('{: <20}  min = {:<10.4g}  avg = {:<10.4g}'.format(
            name, min(times), sum(times) / repeat))


if __name__ == '__main__':
    # data = find_slow_seed()
    run_benchmarks()
