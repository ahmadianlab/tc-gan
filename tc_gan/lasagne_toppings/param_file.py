from logging import getLogger
import contextlib

import lasagne
import numpy as np

logger = getLogger(__name__)

version = 1


def class_path(obj):
    cls = type(obj)
    return cls.__module__ + '.' + cls.__name__


def get_all_param_values_as_dict(layer, trainable=True):
    values = lasagne.layers.get_all_param_values(layer, trainable=trainable)
    params = lasagne.layers.get_all_params(layer, trainable=trainable)
    layers = lasagne.layers.get_all_layers(layer)
    dval = {'pval_{}'.format(i): p for i, p in enumerate(values)}
    dval.update(
        version=version,
        param_names=[p.name for p in params],
        layer_classes=list(map(class_path, layers)),
    )
    return dval


def dump(layer, path):
    np.savez_compressed(path, **get_all_param_values_as_dict(layer))


def load(path):
    # TODO: use stored names for validation
    npz = np.load(path)
    try:
        version = int(npz['version'])
    except KeyError:
        version = 0
    return _loaders[version](npz)


def _load_v1(npz):
    keys = [k for k in npz if k.startswith('pval_')]
    keys.sort(key=lambda k: int(k[len('pval_'):]))
    return [npz[k] for k in keys]


def _load_v0(npz):
    _keys, values = zip(*sorted(npz.items(), key=lambda kv: int(kv[0])))
    return list(values)

_loaders = {
    0: _load_v0,
    1: _load_v1,
}


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
            logger.info('An error occurred. Saving parameters to: %s, %s',
                        pre_path, post_path)

            np.savez_compressed(pre_path, **values)
            np.savez_compressed(post_path,
                                **get_all_param_values_as_dict(layer))

            logger.info('Saved parameters to: %s, %s',
                        pre_path, post_path)


def wrap_with_save_on_error(layer, pre_path, post_path):
    def decorator(func):
        def wrapper(*args, **kwds):
            with save_on_error(layer, pre_path, post_path):
                return func(*args, **kwds)
        return wrapper
    return decorator
