import numpy as np

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


def test_rate_to_volt(k=ssnode.DEFAULT_PARAMS['k'],
                      n=ssnode.DEFAULT_PARAMS['n']):
    xs = np.linspace(0, 1000, 1000)
    ys_py = ssnode.rate_to_volt(xs, k=k, n=n)
    ys_c = np.array([libssnode.rate_to_volt(x, k, n) for x in xs])
    np.testing.assert_allclose(ys_py, ys_c, rtol=0, atol=1e-12)
