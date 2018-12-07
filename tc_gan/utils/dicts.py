def subdict_by_prefix(flat, prefix, key=None):
    """
    Put key-value pairs in `flat` dict prefixed by `prefix` in a sub-dict.

    >>> flat = dict(
    ...     prefix_alpha=1,
    ...     prefix_beta=2,
    ...     gamma=3,
    ... )
    >>> assert subdict_by_prefix(flat, 'prefix_') == dict(
    ...     prefix=dict(alpha=1, beta=2),
    ...     gamma=3,
    ... )

    Key of the sub-dictionary can be explicitly specified:

    >>> assert subdict_by_prefix(flat, 'prefix_', 'delta') == dict(
    ...     delta=dict(alpha=1, beta=2),
    ...     gamma=3,
    ... )

    If the sub-dictionary already exists, it is copied and then
    extended:

    >>> flat['prefix'] = dict(theta=4)
    >>> assert subdict_by_prefix(flat, 'prefix_') == dict(
    ...     prefix=dict(alpha=1, beta=2, theta=4),
    ...     gamma=3,
    ... )
    >>> assert flat['prefix'] == dict(theta=4)  # i.e., not modified

    """
    if key is None:
        key = prefix.rstrip('_')
    nested = {}
    nested[key] = subdict = flat.get(key, {}).copy()
    assert isinstance(subdict, dict)

    for k, v in flat.items():
        if k == key:
            pass
        elif k.startswith(prefix):
            subdict[k[len(prefix):]] = v
        else:
            nested[k] = v

    return nested


def iteritemsdeep(dct):
    """
    Works like ``dict.iteritems`` but iterate over all descendant items

    >>> dct = dict(a=1, b=2, c=dict(d=3, e=4))
    >>> sorted(iteritemsdeep(dct))
    [(('a',), 1), (('b',), 2), (('c', 'd'), 3), (('c', 'e'), 4)]

    """
    for (key, val) in dct.items():
        if isinstance(val, dict):
            for (key_child, val_child) in iteritemsdeep(val):
                yield ((key,) + key_child, val_child)
        else:
            yield ((key,), val)
# Taken from dictsdiff.core


def getdeep(dct, key):
    """
    Get deeply nested value of a dict-like object `dct`.

    >>> dct = {'a': {'b': {'c': 1}}}
    >>> getdeep(dct, 'a.b.c')
    1
    >>> getdeep(dct, 'a.b.d')
    Traceback (most recent call last):
      ...
    KeyError: 'd'

    """
    if not isinstance(key, tuple):
        key = key.split('.')
    for k in key[:-1]:
        dct = dct[k]
    return dct[key[-1]]
