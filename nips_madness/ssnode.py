from __future__ import print_function, division

try:
    range = xrange  # Python 2/3 compatibility
except NameError:
    pass

from contextlib import contextmanager
import collections
import itertools
import time

import numpy
import scipy.optimize

from .clib import libssnode, double_ptr


class FixedPointResult(object):

    message = None

    def __init__(self, x, error):
        self.x = x
        self.error = error

    @property
    def success(self):
        return self.error == 0

    def to_exception(self):
        return FixedPointError(self.message, self)


class FixedPointError(Exception):

    def __init__(self, message, result):
        super(FixedPointError, self).__init__(message)
        self.result = result


@contextmanager
def print_timing(header=''):
    pre = time.time()
    yield
    t = time.time() - pre
    print(header, t)


def take(n, iterable):
    """
    Return first n items of the iterable as a list

    from: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    return list(itertools.islice(iterable, n))


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


def fixed_point_equation(r, ext, W, k, n, io_fun):
    return -r + io_fun(numpy.dot(W, r) + ext)


def drdt(r, _t, ext, W, k, n, io_fun, tau):
    return fixed_point_equation(r, ext, W, k, n, io_fun) / tau


def fixed_point_root(W, ext, r0, k, n, root_kwargs={}, **kwds):
    """
    Find the fixed point of the SSN and return `OptimizeResult` object.

    Actual state of the SSN can be accessed via `.x`; i.e.,::

        sol = fixed_point(W, ext)
        fp_state = sol.x

    """
    io_fun = make_io_fun(k=k, n=n, **kwds)
    args = (ext, W, k, n, io_fun)
    return scipy.optimize.root(fixed_point_equation, r0, args, **root_kwargs)


def rate_to_volt(rate, k, n):
    return (rate / k)**(1 / n)


def io_alin(v, volt_max, k, n):
    vc = numpy.clip(v, 0, volt_max)
    rate = k * (vc**n)
    linear = k * (volt_max**(n-1)) * n * (v - volt_max)
    return numpy.where(v <= volt_max, rate, rate + linear)


def io_power(v, k, n):
    rate = k * (thlin(v)**n)
    return rate


def io_atanh(v, r0, r1, v0, k, n):
    v_pow = numpy.clip(v, 0, v0)
    r_pow = k * (v_pow**n)
    r_tanh = r0 + (r1 - r0) * numpy.tanh(
        n * r0 / (r1 - r0) * (v - v0) / v0
    )
    return numpy.where(v <= v0, r_pow, r_tanh)


def solve_dynamics(*args, **kwds):
    sol = fixed_point(*args, **kwds)
    if not sol.success:
        print(sol.success)
    return sol.x


def fixed_point(
        W, ext, k, n, r0, tau=[.016, .002],
        # max_iter=300, atol=1e-8, dt=.0001, solver='gsl',
        max_iter=10000, atol=1e-10, dt=.001, solver='euler',
        rate_soft_bound=100, rate_hard_bound=200,
        rate_stop_at=numpy.inf,
        io_type='asym_tanh', check=False):
    """
    Solve ODE for the SSN until it converges to a fixed point.

    Note: if ``solver='euler'``, it may be better to use ``atol=1e-10``
    and ``dt=.001``.

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
        Absolute tolerance to the change in rate.
        If ``solver='gsl'``, the error is measured in terms of the
        component-wise difference between the current rate and 0.1
        seconds past rate.
        If ``solver='euler'``, `dt` is used instead of 0.1.
    rate_soft_bound : float
        The I/O function is power-law below this point.
    rate_hard_bound : float
        The true maximum rate.  Used only when ``io_type='asym_tanh'``.
    io_type : {'asym_linear', 'asym_tanh'}
        If ``'asym_linear'`` (default), the I/O function is linear after
        `rate_soft_bound`.  If ``'asym_tanh'``, the I/O function is
        tanh after `rate_soft_bound` and the rate is bounded by
        `rate_hard_bound`.
    solver : {'gsl', 'euler'}
        ODE solver to be used.
        gsl uses "msadams" solver implemented in the GNU Scientific
        Library (A variable-coefficient linear multistep Adams
        method).
        euler is the hand-coded C implementation.
    check : bool
        Raise `FixedPointError` if convergence failed.  Default is `False`.

    Returns
    -------
    sol : FixedPointResult
        It is an object with the following attributes:

        x : array of shape (2*N,)
            A fixed point (or something else if the ODE is failed to
            converge).
        success : bool
            Whether or not a fixed point is found.
        message : str
            Error/success message.

    """
    if io_type not in ('asym_linear', 'asym_tanh'):
        raise ValueError("Unknown I/O type: {}".format(io_type))
    if solver not in ('gsl', 'euler'):
        raise ValueError("Unknown solver: {}".format(solver))

    W = numpy.asarray(W, dtype='double')
    ext = numpy.asarray(ext, dtype='double')
    r0 = numpy.array(r0, dtype='double')  # copied, as it will be modified
    r1 = numpy.empty_like(r0)
    tau_E, tau_I = tau

    N = W.shape[0] // 2
    assert 2 * N == W.shape[0] == W.shape[1]
    assert W.ndim == 2
    assert (2 * N,) == r0.shape == ext.shape

    if io_type == 'asym_linear':
        rate_hard_bound = rate_stop_at

    error = getattr(libssnode,
                    'solve_dynamics_{}_{}'.format(io_type, solver))(
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
    sol = FixedPointResult(r0, error)
    if error == 0:
        if numpy.isfinite(r0).all():
            sol.message = "Converged"
        else:
            sol.error = 1
            sol.message = "Converged to non-finite value"
    elif error == 1:
        sol.message = "SSN Convergence Failed"
    elif error == 2:
        sol.message = "Reached to rate_stop_at"
    elif error > 900:
        sol.message = "GSL error {}".format(error - 1000)
    else:
        sol.message = "Unknown error: code={}".format(error)
    if check and not sol.success:
        raise sol.to_exception()
    return sol


def make_io_fun(k, n,
                rate_soft_bound=100, rate_hard_bound=200,
                io_type='asym_tanh'):
    v0 = rate_to_volt(rate_soft_bound, k, n)
    if io_type == 'asym_linear':
        def io_fun(v):
            return io_alin(v, v0, k, n)
    elif io_type == 'asym_tanh':
        def io_fun(v):
            return io_atanh(v, rate_soft_bound, rate_hard_bound, v0, k, n)
    elif io_type == 'asym_power':
        def io_fun(v):
            return io_power(v, k, n)
    else:
        raise ValueError("Unknown I/O type: {}".format(io_type))
    return io_fun


def solve_dynamics_python(
        W, ext, k, n, r0, tau=[.016, .002],
        max_iter=100000, atol=1e-10, dt=.001,
        **kwds):

    io_fun = make_io_fun(k=k, n=n, **kwds)
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


def odeint(t, W, ext, r0, k, n, tau=[.016, .002],
           odeint_kwargs={}, **kwds):
    io_fun = make_io_fun(k=k, n=n, **kwds)
    N = W.shape[0] // 2
    tau = any_to_neu_vec(N, tau)
    args = (ext, W, k, n, io_fun, tau)
    return scipy.integrate.odeint(drdt, r0, t, args, **odeint_kwargs)


FixedPointsInfo = collections.namedtuple('FixedPointsInfo', [
    'solutions', 'counter', 'rejections',
])


def find_fixed_points(num, Z_W_gen, exts, **common_kwargs):
    exts = list(exts)  # make sure it can be iterated multiple times
    counter = collections.Counter()

    def infinite_solutions():
        for Z, W in Z_W_gen:
            solutions = []
            for ext in exts:
                sol = fixed_point(W, ext, **common_kwargs)
                if not sol.success:
                    counter[sol.error] += 1
                    break
                solutions.append(sol)
            else:
                yield Z, [s.x for s in solutions], solutions

    zs, xs, solutions = zip(*take(num, infinite_solutions()))
    return zs, xs, FixedPointsInfo(
        solutions,
        counter,
        sum(counter.values()),
    )


def plot_io_funs(k=0.01, n=2.2, r0=100, r1=200, xmin=-1, xmax=150):
    from matplotlib import pyplot

    fig, ax = pyplot.subplots()
    x = numpy.linspace(xmin, xmax)
    v0 = rate_to_volt(r0, k, n)
    ax.plot(x, k * (thlin(x)**n), label='power-law')
    ax.plot(x, io_alin(x, v0, k, n), label='asym_linear')
    ax.plot(x, io_atanh(x, r0, r1, v0, k, n), label='asym_tanh')

    ax.legend(loc='best')


def plot_trajectory(
        N=102,
        J=numpy.array([[.0957, .0638], [.1197, .0479]]),
        D=numpy.array([[.7660, .5106], [.9575, .3830]]),
        S=numpy.array([[.6667, .2], [1.333, .2]]) / 8,
        # io_type='asym_linear', seed=97,
        io_type='asym_tanh', seed=65,
        bandwidth=1, smoothness=0.25/8, contrast=20,
        fp_solver_kwargs={},
        tmax=3, tnum=300, plot_fp=False):
    import stimuli
    from .tests.test_dynamics import numeric_w
    from matplotlib import pyplot

    np = numpy
    J = np.array([[.0957, .0638], [.1197, .0479]])
    D = np.array([[.7660, .5106], [.9575, .3830]])
    S = np.array([[.6667, .2], [1.333, .2]]) / 8

    if io_type == 'asym_linear':
        # Boost inhibition if io_type=='asym_linear'; otherwise the
        # activity diverges.
        J[:, 1] *= 1.7

    rs = numpy.random.RandomState(seed)
    Z = rs.rand(1, 2*N, 2*N)
    W, = numeric_w(Z, J, D, S)
    X = numpy.linspace(-0.5, 0.5, N)
    ext, = stimuli.input([bandwidth], X, smoothness, contrast)

    solver_kwargs = dict(
        W=W,
        ext=ext,
        r0=numpy.zeros(W.shape[0]),
        k=0.01, n=2.2,
        io_type=io_type,
    )

    t = numpy.linspace(0, tmax, tnum)
    with print_timing('Runtime (odeint):'):
        r = odeint(t, **solver_kwargs)

    if plot_fp:
        # with print_timing('Runtime (fixed_point):'):
        #     sol = fixed_point(**dict(solver_kwargs, **fp_solver_kwargs))
        # fp = sol.x
        with print_timing('Runtime (solve_dynamics):'):
            fp = solve_dynamics(**dict(solver_kwargs, **fp_solver_kwargs))
        print("max |odeint - fp_solver| =", abs(r[-1] - fp).max())

    fig, (ax1, ax2) = pyplot.subplots(nrows=2, sharex=True)
    ax1.plot(t, r[:, :N], color='C0', linewidth=0.5)
    ax1.set_ylabel('Excitatory neurons')

    ax2.plot(t, r[:, N:], color='C1', linewidth=0.5)
    ax2.set_ylabel('Inhibitory neurons')
    ax2.set_xlabel('Time')

    if plot_fp:
        ax1.plot(tmax, [fp[:N]], 'o', color='C0')
        ax2.plot(tmax, [fp[N:]], 'o', color='C1')

    return locals()
