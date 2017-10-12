import contextlib

import lasagne
import numpy as np


def get_all_param_values_as_dict(layer, trainable=True):
    values = lasagne.layers.get_all_param_values(layer, trainable=trainable)
    return {str(i): p for i, p in enumerate(values)}


def dump(layer, path):
    np.savez_compressed(path, **get_all_param_values_as_dict(layer))


def load(path):
    npz = np.load(path)
    _keys, values = zip(*sorted(npz.items(), key=lambda kv: int(kv[0])))
    return list(values)


@contextlib.contextmanager
def save_on_error(layer, pre_path, post_path):
    """Save parameter of `layer` upon an exception."""
    haserror = True
    values = get_all_param_values_as_dict(layer)
    try:
        yield
        haserror = False
    finally:
        if haserror:
            np.savez_compressed(pre_path, **values)
            np.savez_compressed(post_path,
                                **get_all_param_values_as_dict(layer))


def wrap_with_save_on_error(layer, pre_path, post_path):
    def decorator(func):
        def wrapper(*args, **kwds):
            with save_on_error(layer, pre_path, post_path):
                return func(*args, **kwds)
        return wrapper
    return decorator
