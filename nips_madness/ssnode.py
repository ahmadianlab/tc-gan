from __future__ import print_function, division

import numpy
import scipy.optimize

from .weight_gen import generate_parameter


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


def solve_dynamics(t, W, ext, r0=None, k=0.04, n=2, tau=[1, 0.1], **kwds):
    if r0 is None:
        r0 = thlin(numpy.linalg.solve(W, -ext))
    N = W.shape[0] // 2
    tau = any_to_neu_vec(N, tau)
    args = (ext, W, k, n, tau)
    return scipy.integrate.odeint(drdt, r0, t, args, **kwds)


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


def demo_dynamics(N=50,
                  J=[[1, 3], [1, 1]],
                  delta=numpy.ones((2, 2)) * 0.1,
                  sigma=numpy.ones((2, 2)) * 0.2,
                  ext=[40, 20],
                  tmax=3, tnum=300, seed=0, plot_fp=False):
    from matplotlib import pyplot
    W, _ = generate_parameter(N, J, delta, sigma, seed=seed)
    ext = any_to_neu_vec(N, ext)
    t = numpy.linspace(0, tmax, tnum)
    r = solve_dynamics(t, W, ext)

    fig, (ax1, ax2) = pyplot.subplots(nrows=2, sharex=True)
    ax1.plot(t, r[:, :N], color='C0', linewidth=0.5)
    ax1.set_ylabel('Excitatory neurons')

    ax2.plot(t, r[:, N:], color='C1', linewidth=0.5)
    ax2.set_ylabel('Inhibitory neurons')
    ax2.set_xlabel('Time')

    if plot_fp:
        sol = fixed_point(W, ext)
        ax1.plot(tmax, [sol.x[:N]], 'o', color='C0')
        ax2.plot(tmax, [sol.x[N:]], 'o', color='C1')

        if not sol.success:
            print(sol.message)
