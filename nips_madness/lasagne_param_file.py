import contextlib

import lasagne
import numpy as np


def get_all_param_values_as_dict(layer):
    return {str(i): p for i, p in
            enumerate(lasagne.layers.get_all_param_values(layer))}


def dump(layer, path):
    np.savez_compressed(path, **get_all_param_values_as_dict(layer))


def load(path):
    npz = np.load(path)
    _keys, values = zip(*sorted(npz.items(), key=lambda kv: int(kv[0])))
    return list(values)


@contextlib.contextmanager
def save_on_error(layer, path):
    """Save parameter of `layer` to `path` upon an exception."""
    haserror = True
    values = get_all_param_values_as_dict(layer)
    try:
        yield
        haserror = False
    finally:
        if haserror:
            np.savez_compressed(path, **values)


def wrap_with_save_on_error(layer, path):
    def decorator(func):
        def wrapper(*args, **kwds):
            with save_on_error(layer, path):
                return func(*args, **kwds)
        return wrapper
    return decorator
