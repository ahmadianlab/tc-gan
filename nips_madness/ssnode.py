from __future__ import print_function, division

try:
    range = xrange  # Python 2/3 compatibility
except NameError:
    pass

from contextlib import contextmanager
import collections
import itertools
import multiprocessing.dummy
import time

try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import scipy.optimize

from .clib import libssnode, double_ptr
from .gradient_expressions.utils import subsample_neurons


DEFAULT_PARAMS = dict(
    N=102,
    J=np.array([[.0957, .0638], [.1197, .0479]]),
    D=np.array([[.7660, .5106], [.9575, .3830]]),
    S=np.array([[.6667, .2], [1.333, .2]]) / 8,
    bandwidths=[0, 0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1],
    smoothness=0.25/8,
    contrast=[20],
    io_type='asym_tanh',
    k=0.01,
    n=2.2,
    rate_soft_bound=200, rate_hard_bound=1000,
    tau=(0.01589, 0.002),  # Superstition: integer ratio tau_E/I is bad.
)


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
    return np.array([E] * N + [I] * N)


def any_to_neu_vec(N, vec):
    vec = np.asarray(vec)
    if len(vec) == 2:
        vec = make_neu_vec(N, *vec)
    return vec


def thlin(x):
    return x * (x > 0)


def fixed_point_equation(r, ext, W, k, n, io_fun):
    return -r + io_fun(np.dot(W, r) + ext)


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
    vc = np.clip(v, 0, volt_max)
    rate = k * (vc**n)
    linear = k * (volt_max**(n-1)) * n * (v - volt_max)
    return np.where(v <= volt_max, rate, rate + linear)


def io_power(v, k, n):
    rate = k * (thlin(v)**n)
    return rate


def io_atanh(v, r0, r1, v0, k, n):
    v_pow = np.clip(v, 0, v0)
    r_pow = k * (v_pow**n)
    r_tanh = r0 + (r1 - r0) * np.tanh(
        n * r0 / (r1 - r0) * (v - v0) / v0
    )
    return np.where(v <= v0, r_pow, r_tanh)


def solve_dynamics(*args, **kwds):
    sol = fixed_point(*args, **kwds)
    if not sol.success:
        print(sol.message)
    return sol.x


def fixed_point(
        W, ext, k, n, r0, tau=DEFAULT_PARAMS['tau'],
        max_iter=10000, atol=1e-5, dt=.0008, solver='euler',
        rate_soft_bound=DEFAULT_PARAMS['rate_soft_bound'],
        rate_hard_bound=DEFAULT_PARAMS['rate_hard_bound'],
        rate_stop_at=np.inf,
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
        If ``solver='euler'``, `dt` is used instead of 0.1.
    rate_soft_bound : float
        The I/O function is power-law below this point.
    rate_hard_bound : float
        The true maximum rate.  Used only when ``io_type='asym_tanh'``.
    io_type : {'asym_power', 'asym_linear', 'asym_tanh'}
        If ``'asym_linear'`` (default), the I/O function is linear after
        `rate_soft_bound`.  If ``'asym_tanh'``, the I/O function is
        tanh after `rate_soft_bound` and the rate is bounded by
        `rate_hard_bound`.
    solver : {'euler'}
        ODE solver to be used.
        euler is the hand-coded C implementation.
        At this point 'euler' is the only choice.
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
    if io_type not in ('asym_linear', 'asym_tanh', 'asym_power'):
        raise ValueError("Unknown I/O type: {}".format(io_type))
    if solver not in ('euler'):
        raise ValueError("Unknown solver: {}".format(solver))

    W = np.asarray(W, dtype='double')
    ext = np.asarray(ext, dtype='double')
    r0 = np.array(r0, dtype='double')  # copied, as it will be modified
    r1 = np.empty_like(r0)
    tau_E, tau_I = tau

    N = W.shape[0] // 2
    assert 2 * N == W.shape[0] == W.shape[1]
    assert W.ndim == 2
    assert (2 * N,) == r0.shape == ext.shape

    if io_type in ('asym_power', 'asym_linear'):
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
        if np.isfinite(r0).all():
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
                rate_soft_bound=DEFAULT_PARAMS['rate_soft_bound'],
                rate_hard_bound=DEFAULT_PARAMS['rate_hard_bound'],
                io_type=DEFAULT_PARAMS['io_type']):
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
        W, ext, k, n, r0, tau=DEFAULT_PARAMS['tau'],
        max_iter=100000, atol=1e-10, dt=.001,
        **kwds):

    io_fun = make_io_fun(k=k, n=n, **kwds)
    N = W.shape[0] // 2
    tau = any_to_neu_vec(N, tau)

    rr = r0

    for _ in range(max_iter):
        dr = (- rr + io_fun(np.dot(W, rr) + ext))/tau
        rr = rr + dt * dr
        if np.abs(dr).max() < atol:
            break
    else:
        print("SSN Convergence Failed")

    return rr


def odeint(t, W, ext, r0, k, n, tau=DEFAULT_PARAMS['tau'],
           odeint_kwargs={}, **kwds):
    io_fun = make_io_fun(k=k, n=n, **kwds)
    N = W.shape[0] // 2
    tau = any_to_neu_vec(N, tau)
    args = (ext, W, k, n, io_fun, tau)
    return scipy.integrate.odeint(drdt, r0, t, args, **odeint_kwargs)


FixedPointsInfo = collections.namedtuple('FixedPointsInfo', [
    'solutions', 'counter', 'rejections', 'unused',
])
null_FixedPointsInfo = FixedPointsInfo(None, None, 0, 0)


def find_fixed_points(num, Z_W_gen, exts, method='parallel', **common_kwargs):
    """
    Find `num` sets of fixed points using weight matrices from `Z_W_gen`.

    Fixed points are calculated for each external input in `exts`.  A
    set of fixed points is returned only if the fixed points are found
    for all external inputs.

    Parameters
    ----------
    num : int
        Number of the sets of fixed points.
    Z_W_gen : iterable
        An iterable yielding pairs ``(Z, W)``.
    exts : array-like of shape (NB, 2N)
        A list of external inputs.
    method : 'parallel' or 'serial'
        Run the solver using multiple threads if ``'parallel'`` (default).

    Other keyword arguments are passed to `fixed_point`.

    Returns
    -------
    Zs : array of shape (num, ...)
        "List" of Zs yielded with Ws from `Z_W_gen`.
    Rs : array of shape (num, NB, 2N)
        "List" of fixed points.
    info : FixedPointsInfo
        Supplementary information from the solver.  It has the following
        attributes:

        solutions : [FixedPointResult]
            List of solution objects for successful convergence cases.
            The order is same as `Zs` and `Rs`.
        counter : collections.Counter
            A mapping from error code to the number of occurrences.
        rejections : int
            Number of rejections.
        unused : int
            Number of unused solutions.  This value may vary for run
            to run, due to nondeterministic behavior of multi-core
            computation.  Since this function typically requires
            `Z_W_gen` to generate Ws using random number generator,
            this number has to be recorded if one needs to reproduce
            the same computation.

    """
    if method == 'parallel':
        finder = find_fixed_points_parallel
    elif method == 'serial':
        finder = find_fixed_points_serial
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return finder(num, Z_W_gen, exts, **common_kwargs)


def find_fixed_points_serial(num, Z_W_gen, exts, **common_kwargs):
    # Revers exts to try large bandwidth first, as it is more likely
    # to produces diverging solution:
    exts = list(exts)
    exts.reverse()

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
                # Reverse the solution back to the original order:
                solutions.reverse()
                yield Z, [s.x for s in solutions], solutions

    zs, xs, solutions = zip(*take(num, infinite_solutions()))
    zs = np.array(zs)
    xs = np.array(xs)
    return zs, xs, FixedPointsInfo(
        solutions,
        counter,
        sum(counter.values()),
        0,
    )


def find_fixed_points_parallel(num, Z_W_gen, exts, no_pool=False,
                               resubmit_threshold=0,
                               deterministic=True,
                               **common_kwargs):
    # Revers exts to try large bandwidth first, as it is more likely
    # to produces diverging solution:
    exts = list(exts)
    exts.reverse()

    results = queue.Queue(0)
    indices = itertools.count()
    nonlocals = {'consumed': 0}

    def worker(idx, Z, W):
        solutions = []
        try:
            for ext in exts:
                sol = fixed_point(W, ext, **common_kwargs)
                solutions.append(sol)
                if not sol.success:
                    results.put((False, idx, Z, solutions))
                    return
            results.put((True, idx, Z, solutions))
        except Exception as err:
            results.put((err, idx, Z, solutions))

    if no_pool:
        def submit():
            Z, W = next(Z_W_gen)
            worker(next(indices), Z, W)
            nonlocals['consumed'] += 1
    else:
        pool = multiprocessing.dummy.Pool()

        def submit():
            Z, W = next(Z_W_gen)
            worker, next(indices), Z, W
            pool.apply_async(worker, (next(indices), Z, W))
            nonlocals['consumed'] += 1

    for _ in range(num):
        submit()

    samples = []
    counter = collections.Counter()

    def collect():
        success, idx, Z, solutions = results.get()
        if isinstance(success, Exception):
            raise success
        elif success:
            solutions.reverse()
            samples.append((idx, Z, solutions))
        else:
            counter[solutions[-1].error] += 1
        return success

    while True:
        success = collect()
        if len(samples) >= num:
            break
        if not success or len(samples) <= num * resubmit_threshold:
            try:
                submit()
            except StopIteration:
                break

    unused = nonlocals['consumed'] - num - sum(counter.values())
    if deterministic:
        for _ in range(unused):
            collect()

    samples.sort(key=lambda x: x[0])
    samples = samples[:num]
    _, zs, solutions = zip(*samples)
    xs = [[s.x for s in sols] for sols in solutions]

    if not no_pool:
        pool.terminate()
        pool.join()

    zs = np.array(zs)
    xs = np.array(xs)
    return zs, xs, FixedPointsInfo(
        solutions,
        counter,
        sum(counter.values()),
        unused,
    )


def plot_io_funs(k=0.01, n=2.2, r0=100, r1=200, xmin=-1, xmax=150):
    from matplotlib import pyplot

    fig, ax = pyplot.subplots()
    x = np.linspace(xmin, xmax)
    v0 = rate_to_volt(r0, k, n)
    ax.plot(x, k * (thlin(x)**n), label='power-law')
    ax.plot(x, io_alin(x, v0, k, n), label='asym_linear')
    ax.plot(x, io_atanh(x, r0, r1, v0, k, n), label='asym_tanh')

    ax.legend(loc='best')


def make_solver_params(
        N=DEFAULT_PARAMS['N'],
        J=DEFAULT_PARAMS['J'],
        D=DEFAULT_PARAMS['D'],
        S=DEFAULT_PARAMS['S'],
        io_type=DEFAULT_PARAMS['io_type'],
        seed=65,
        bandwidth=1,
        smoothness=DEFAULT_PARAMS['smoothness'],
        contrast=DEFAULT_PARAMS['contrast'],
        k=DEFAULT_PARAMS['k'],
        n=DEFAULT_PARAMS['n'],
        ):
    from . import stimuli
    from .tests.test_dynamics import numeric_w

    if isinstance(seed, int):
        rs = np.random.RandomState(seed)
    else:
        rs = seed

    Z = rs.rand(1, 2*N, 2*N)
    W, = numeric_w(Z, J, D, S)
    X = np.linspace(-0.5, 0.5, N)
    ext, = stimuli.input([bandwidth], X, smoothness, contrast)

    return dict(
        W=W,
        ext=ext,
        r0=np.zeros(W.shape[0]),
        k=k, n=n,
        io_type=io_type,
    )


def sample_fixed_points(
        NZ=30, seed=0,
        N=DEFAULT_PARAMS['N'],
        J=DEFAULT_PARAMS['J'],
        D=DEFAULT_PARAMS['D'],
        S=DEFAULT_PARAMS['S'],
        bandwidths=DEFAULT_PARAMS['bandwidths'],
        smoothness=DEFAULT_PARAMS['smoothness'],
        contrast=DEFAULT_PARAMS['contrast'],
        io_type=DEFAULT_PARAMS['io_type'],
        k=DEFAULT_PARAMS['k'],
        n=DEFAULT_PARAMS['n'],
        **solver_kwargs):
    from . import stimuli
    from .tests.test_dynamics import numeric_w

    X = np.linspace(-0.5, 0.5, N)
    exts = stimuli.input(bandwidths, X, smoothness, contrast)
    rs = np.random.RandomState(seed)

    def Z_W_gen():
        while True:
            z = rs.rand(1, 2*N, 2*N)
            W, = numeric_w(z, J, D, S)
            yield z[0], W

    solver_kwargs.setdefault('r0', np.zeros(2 * N))
    solver_kwargs.update(k=k, n=n, io_type=io_type)
    return find_fixed_points(NZ, Z_W_gen(), exts, **solver_kwargs)


def sample_tuning_curves(sample_sites=3, track_net_identity=False, **kwargs):
    _, rates, _ = sample = sample_fixed_points(**kwargs)
    rates = np.array(rates)
    tunings = subsample_neurons(rates, sample_sites,
                                track_net_identity=track_net_identity).T
    return tunings, sample


def plot_trajectory(
        fp_solver_kwargs={},
        tmax=3, tnum=300, plot_fp=False,
        **modelparams):
    from matplotlib import pyplot

    solver_kwargs = make_solver_params(**modelparams)
    N = solver_kwargs['W'].shape[0] // 2

    t = np.linspace(0, tmax, tnum)
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
