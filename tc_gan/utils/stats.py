import numpy


def getstat(thing, stat):
    if stat == 'snr':  # signal-to-noise ratio
        m = thing.mean()
        s = thing.std()
        return m / s
    elif stat == 'cv':
        m = thing.mean()
        s = thing.std()
        return s / m
    elif stat == 'fano':
        m = thing.mean()
        v = thing.var()
        return v / m

    return getattr(thing, stat)()


def rolling_stat(stat, x, window):
    """
    Calculate rolling statistics `stat` of `x` with sliding `window`.

    Parameters
    ----------
    stat : 'snr', 'cv', 'fano', 'mean', 'median', ...
        Name (`str`) of the rolling statistics supported by pandas
        and `getstat` (e.g., `'snr'`).
        https://pandas.pydata.org/pandas-docs/stable/api.html#standard-moving-window-functions
    x : array-like
        1D or 2D numpy array or anything convertable to it.
    window : int
        Size of window.

    Examples
    --------
    >>> rolling_stat('mean', [1, 2, 3, 4], 2)
    array([ 1.5,  2.5,  3.5])
    >>> rolling_stat('median', [[1, 5], [2, 6], [3, 7], [4, 8]], 3)
    array([[ 2.,  6.],
           [ 3.,  7.]])

    """
    import pandas
    if window < 2:
        return x
    x = numpy.asarray(x)
    if x.ndim == 1:
        z = pandas.Series(x)
    else:
        z = pandas.DataFrame(x)
    y = getstat(z.rolling(window=window), stat).values
    return y[window - 1:]
