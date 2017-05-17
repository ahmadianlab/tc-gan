import numpy as np
import pytest

from ..clib import libssnode
from .. import ssnode


def check_io_fun(io_type, io_fun_c,
                 r0=ssnode.DEFAULT_PARAMS['rate_soft_bound'],
                 r1=ssnode.DEFAULT_PARAMS['rate_hard_bound'],
                 k=ssnode.DEFAULT_PARAMS['k'],
                 n=ssnode.DEFAULT_PARAMS['n']):
    v0 = ssnode.rate_to_volt(r0, k=k, n=n)
    xs = np.linspace(-0.1, v0 * 3, 1000)
    io_fun = ssnode.make_io_fun(
        k=k, n=n, rate_soft_bound=r0, rate_hard_bound=r1,
        io_type=io_type)
    ys_py = io_fun(xs)
    ys_c = np.array([io_fun_c(x, r0, r1, v0, k, n) for x in xs])
    np.testing.assert_allclose(ys_py, ys_c, rtol=0, atol=1e-12)


def test_io_atanh():
    check_io_fun('asym_tanh', libssnode.io_atanh)


def test_io_alin():
    check_io_fun('asym_linear', libssnode.io_alin)


def test_io_power():
    check_io_fun('asym_power', libssnode.io_pow)


def test_io_sub_bound(r0=ssnode.DEFAULT_PARAMS['rate_soft_bound'],
                      r1=ssnode.DEFAULT_PARAMS['rate_hard_bound'],
                      k=ssnode.DEFAULT_PARAMS['k'],
                      n=ssnode.DEFAULT_PARAMS['n']):
    common_kwargs = dict(
        k=k, n=n, rate_soft_bound=r0, rate_hard_bound=r1)
    v0 = ssnode.rate_to_volt(r0, k=k, n=n)
    io_pow = ssnode.make_io_fun(io_type='asym_power', **common_kwargs)
    io_atanh = ssnode.make_io_fun(io_type='asym_tanh', **common_kwargs)
    io_alin = ssnode.make_io_fun(io_type='asym_linear', **common_kwargs)
    assert 0 == io_pow(0) == io_atanh(0) == io_alin(0)
    assert io_pow(r0 + 10) != io_atanh(r0 + 10)
    assert io_pow(r0 + 10) != io_alin(r0 + 10)

    for xs in [np.linspace(-10, 1000, 1000), [-0.0, 0, v0]]:
        xs = np.array(xs)
        ys_pow = io_pow(xs)
        ys_atanh = io_pow(xs)
        ys_alin = io_pow(xs)
        np.testing.assert_allclose(ys_pow, ys_atanh, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ys_pow, ys_alin, rtol=0, atol=1e-12)


def test_rate_to_volt(k=ssnode.DEFAULT_PARAMS['k'],
                      n=ssnode.DEFAULT_PARAMS['n']):
    xs = np.linspace(0, 1000, 1000)
    ys_py = ssnode.rate_to_volt(xs, k=k, n=n)
    ys_c = np.array([libssnode.rate_to_volt(x, k, n) for x in xs])
    np.testing.assert_allclose(ys_py, ys_c, rtol=0, atol=1e-12)


@pytest.mark.parametrize('seed', range(10))
def test_fixed_point_c_vs_py(seed):
    kwargs = ssnode.make_solver_params(seed=seed, io_type='asym_tanh')
    kwargs.update(atol=1e-10, tau=(.016, .002))
    sol = ssnode.fixed_point(**kwargs)
    kwargs2 = dict(kwargs, r0=sol.x)
    xpy = ssnode.solve_dynamics_python(**kwargs2)
    np.testing.assert_allclose(sol.x, xpy, rtol=0, atol=1e-7)
