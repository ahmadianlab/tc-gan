from __future__ import print_function, division

try:
    range = xrange  # Python 2/3 compatibility
except NameError:
    pass

import numpy
import scipy.optimize
import scipy.sparse

from .clib import libssnode, double_ptr


def make_neu_vec(N, E, I):
    """
    Create 2N-dim vector (neural vector) from E/I population-level quantity.
    """
    return numpy.array([E] * N + [I] * N)


def any_to_neu_vec(N, vec):
    vec = numpy.asarray(vec)
    if len(vec) == 2:
        vec = make_neu_vec(N, *vec)
    return vec


def thlin(x):
    return x * (x > 0)


def fixed_point_equation(r, ext, W, k, n):
    return -r + k * thlin(numpy.dot(W, r) + ext)**n


def drdt(r, _t, ext, W, k, n, tau):
    return fixed_point_equation(r, ext, W, k, n) / tau


def fixed_point(W, ext, r0=None, k=0.04, n=2, **kwds):
    """
    Find the fixed point of the SSN and return `OptimizeResult` object.

    Actual state of the SSN can be accessed via `.x`; i.e.,::

        sol = fixed_point(W, ext)
        fp_state = sol.x

    """
    if r0 is None:
        r0 = thlin(numpy.linalg.solve(W, -ext))
    args = (ext, W, k, n)
    return scipy.optimize.root(fixed_point_equation, r0, args, **kwds)


def rate_to_volt(rate, k, n):
    return (rate / k)**(1 / n)


def io_plin(v, volt_max, k, n):
    vc = numpy.clip(v, 0, volt_max)
    rate = k * (vc**n)
    linear = k * (volt_max**(n-1)) * n * (v - volt_max)
    return numpy.where(v <= volt_max, rate, rate + linear)


def io_atanh(v, r0, r1, v0, v1, k, n):
    v_pow = numpy.clip(v, 0, v0)
    r_pow = k * (v_pow**n)
    r_tanh = r0 + (r1 - r0) * numpy.tanh(
        n * r0 / (r1 - r0) * (v - v0) / v0
    )
    return numpy.where(v <= v0, r_pow, r_tanh)


def solve_dynamics(
        W, ext, k, n, r0, tau=[.016, .002],
        max_iter=100000, atol=1e-10, dt=.001,
        rate_soft_bound=100, rate_hard_bound=200,
        io_type='asym_linear'):
    """
    Solve ODE for the SSN until it converges to a fixed point.

    Parameters
    ----------
    W : array of shape (2*N, 2*N)
        The weight matrix.
    ext : array of shape (2*N,)
        Stationary input to the network.
    k : float
        Scaling factor of the I/O function.
    n : float
        Power of the SSN nonlinearity.
    r0 : array of shape (2*N,)
        The initial condition in terms of rate.
    tau : array of shape (2,)
        The time constants of the neurons.
    max_iter : int
        The hard bound to the number of iteration of the Euler method.
    atol : float
        Absolute tolerance to the each component of the derivative.
    rate_soft_bound : float
        The I/O function is power-law below this point.
    rate_hard_bound : float
        The true maximum rate.  Used only when ``io_type='asym_tanh'``.
    io_type : {'asym_linear', 'asym_tanh'}
        If ``'asym_linear'`` (default), the I/O function is linear after
        `rate_soft_bound`.  If ``'asym_tanh'``, the I/O function is
        tanh after `rate_soft_bound` and the rate is bounded by
        `rate_hard_bound`.

    Returns
    -------
    r : array of shape (2*N,)
        A fixed point or something else if the ODE is failed to converge.

    """
    if io_type not in ('asym_linear', 'asym_tanh'):
        raise ValueError("Unknown I/O type: {}".format(io_type))

    W = numpy.asarray(W, dtype='double')
    ext = numpy.asarray(ext, dtype='double')
    r0 = numpy.array(r0, dtype='double')  # copied, as it will be modified
    r1 = numpy.empty_like(r0)
    tau_E, tau_I = tau

    N = W.shape[0] // 2
    assert 2 * N == W.shape[0] == W.shape[1]
    assert W.ndim == 2
    assert (2 * N,) == r0.shape == ext.shape

    error = getattr(libssnode, 'solve_dynamics_{}'.format(io_type))(
        N,
        W.ctypes.data_as(double_ptr),
        ext.ctypes.data_as(double_ptr),
        float(k), float(n),
        r0.ctypes.data_as(double_ptr),
        r1.ctypes.data_as(double_ptr),
        tau_E, tau_I,
        dt, max_iter, atol,
        rate_soft_bound, rate_hard_bound,
    )
    if error == 1:
        print("SSN Convergence Failed")
    return r0


def solve_dynamics_python(
        W, ext, k, n, r0, tau=[.016, .002],
        max_iter=100000, atol=1e-10, dt=.001,
        rate_soft_bound=100, rate_hard_bound=200,
        io_type='asym_linear'):
    v0 = rate_to_volt(rate_soft_bound, k, n)
    v1 = rate_to_volt(rate_hard_bound, k, n)
    if io_type == 'asym_linear':
        def io_fun(v):
            return io_plin(v, v0, k, n)
    elif io_type == 'asym_tanh':
        def io_fun(v):
            return io_atanh(v, rate_soft_bound, rate_hard_bound, v0, v1, k, n)
    else:
        raise ValueError("Unknown I/O type: {}".format(io_type))

    N = W.shape[0] // 2
    tau = any_to_neu_vec(N, tau)

    rr = r0

    for _ in range(max_iter):
        dr = (- rr + io_fun(numpy.dot(W, rr) + ext))/tau
        rr = rr + dt * dr
        if numpy.abs(dr).max() < atol:
            break
    else:
        print("SSN Convergence Failed")

    return rr


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def single_input(N, bandwidth, smoothness=0.01):
    fs = numpy.linspace(-0.5, 0.5, N)
    x = bandwidth / 2 + fs
    y = bandwidth / 2 - fs
    return sigmoid(x / smoothness) * sigmoid(y / smoothness)


def all_inputs(N, bandwidths, **kwds):
    """
    Return an array of inputs in the shape ``(N, len(bandwidths))``.
    """
    return numpy.array([single_input(N, b, **kwds) for b in bandwidths])


def plot_io_funs():
    from matplotlib import pyplot

    fig, ax = pyplot.subplots()
    x = numpy.linspace(-1, 100)
    k = 0.04
    n = 2.2
    r0 = 100
    r1 = 200
    v0 = rate_to_volt(r0, k, n)
    v1 = rate_to_volt(r1, k, n)
    ax.plot(x, k * (thlin(x)**n), label='power-law')
    ax.plot(x, io_plin(x, v0, k, n), label='asym_linear')
    ax.plot(x, io_atanh(x, r0, r1, v0, v1, k, n), label='asym_tanh')

    ax.legend(loc='best')
