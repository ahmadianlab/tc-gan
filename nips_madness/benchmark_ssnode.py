import timeit

import numpy as np

import stimuli
from .ssnode import solve_dynamics, solve_dynamics_python
from .tests.test_dynamics import numeric_w


def make_bench_solve_dynamics(
        fun=solve_dynamics,
        bandwidth=8, smoothness=0.25/8, contrast=20,
        N=51, k=0.01, n=2.2, io_type='asym_linear', **kwds):
    J = np.array([[.0957, .0638], [.1197, .0479]])
    D = np.array([[.7660, .5106], [.9575, .3830]])
    S = np.array([[.6667, .2], [1.333, .2]]) / 8
    if io_type == 'asym_linear':
        # Boost inhibition if io_type=='asym_linear'; otherwise the
        # activity diverges.
        J[:, 1] *= 1.7

    np.random.seed(0)
    Z = np.random.rand(1, 2*N, 2*N)
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
    run_benchmarks()