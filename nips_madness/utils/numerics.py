import warnings

import numpy as np
import theano


def tolist_if_not(arr):
    """
    Convert `arr` to list if it is not already.

    >>> import numpy as np
    >>> tolist_if_not([0])
    [0]
    >>> tolist_if_not(np.arange(1))
    [0]

    """
    try:
        tolist = arr.tolist
    except AttributeError:
        return arr
    return tolist()


def cartesian_product(*arrays, **kwargs):
    """
    Return Cartesian product of `arrays` (`itertools.product` for Numpy).

    >>> cartesian_product([0, 1], [10, 20], dtype=int)
    array([[ 0,  0,  1,  1],
           [10, 20, 10, 20]])

    """
    dtype = kwargs.pop('dtype', theano.config.floatX)
    assert not kwargs
    arrays = list(map(np.asarray, arrays))
    assert all(a.ndim == 1 for a in arrays)

    prod = np.zeros([len(arrays)] + list(map(len, arrays)),
                    dtype=dtype)

    for i, a in enumerate(arrays):
        shape = [1] * len(arrays)
        shape[i] = -1
        prod[i] = a.reshape(shape)

    return prod.reshape((len(arrays), -1))


def as_randomstate(seed):
    if hasattr(seed, 'seed'):
        return seed  # suppose it's a RandomState instance
    else:
        return np.random.RandomState(seed)


def random_minibatches(batchsize, data, strict=False, seed=0):
    num_batches = len(data) // batchsize
    if batchsize > len(data):
        raise ValueError('batchsize = {} > len(data) = {}'
                         .format(batchsize, len(data)))
    if len(data) % batchsize != 0:
        msg = 'len(data) = {} not divisible by batchsize = {}'.format(
            len(data), batchsize)
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)

    rng = as_randomstate(seed)

    def iterator():
        while True:
            idx = np.arange(len(data))
            rng.shuffle(idx)

            for i in range(num_batches):
                s = i * batchsize
                e = (i + 1) * batchsize
                yield data[idx[s:e]]

    return iterator()


def report_allclose_tols(a, b, rtols, atols, **isclose_kwargs):
    """
    Print %mismatch of `a` and `b` with combinations of `rtols` and `atols`.
    """
    for rtol in rtols:
        for atol in atols:
            p = np.isclose(a, b, rtol=rtol, atol=atol, **isclose_kwargs).mean()
            print('mismatch {:>7.3%} with rtol={:<4.1e} atol={:<4.1e}'
                  .format(1 - p, rtol, atol))
