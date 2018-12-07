import numpy as np

from ..utils import rolling_stat


def pw1e(gp, shift, window, rolling='mean'):
    """Point-wise l1 error."""
    d = rolling_stat(rolling, gp[shift:] - gp[:-shift], window)
    return np.abs(d).mean(axis=1)


def smr1e(gp, shift, window, rolling='mean'):
    """Symmetric mean relative l1 error (= sMAPE/200)."""
    from numpy import abs
    d = rolling_stat(rolling, gp[shift:] - gp[:-shift], window)
    s = rolling_stat(rolling, abs(gp[shift:]) + abs(gp[:-shift]), window) / 2
    s += 1e-3
    return abs(d / s).mean(axis=1)
