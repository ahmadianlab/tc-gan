import numpy


def sign_changings(data):
    """
    Calculate indices of sign-changing locations in 1D array `data`.

    aka zero-crossing

    >>> #             *          *      *  *   # changes
    >>> #       0  1  2   3   4  5  6   7  8   # index
    >>> data = [1, 2, 0, -3, -4, 7, 8, -2, 1]
    >>> sign_changings(data)
    array([2, 5, 7, 8])

    Based on: http://stackoverflow.com/a/21468492

    """
    data = numpy.asanyarray(data)
    pos = data > 0
    npos = data < 0
    idx = ((data[1:] == 0) |
           (pos[:-1] & npos[1:]) |
           (npos[:-1] & pos[1:])).nonzero()[0] + 1
    if len(data) > 0 and data[0] == 0:
        return numpy.append(0, idx)
    else:
        return idx


def first_sign_change(data):
    """
    Return the first sign-changing index in 1D array `data`.

    >>> first_sign_change([1, 1, -1, -1, -1, 1, 1])
    2
    >>> first_sign_change([-1, -1, 1, 1, -1, 1, 1])
    2
    >>> first_sign_change([1, 1, 1, 1])  # returns None
    >>> first_sign_change([1, 1, 0, 1, 1])
    2
    >>> first_sign_change([1, 1, 0, -1, -1])
    2

    """
    idx = sign_changings(data)
    if len(idx) > 0:
        return idx[0]
