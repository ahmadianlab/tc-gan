from logging import getLogger

import numpy as np
import theano

logger = getLogger(__name__)


def theano_function(*args, **kwds):
    # from theano.compile.nanguardmode import NanGuardMode
    # kwds.setdefault('mode', NanGuardMode(
    #     nan_is_error=True, inf_is_error=True, big_is_error=True))
    kwds.setdefault('allow_input_downcast', True)
    # MAYBE: make sure to use theano.config.floatX everywhere and
    # remove allow_input_downcast.
    return theano.function(*args, **kwds)


def is_theano(a):
    return isinstance(a, theano.Variable)


def get_array_module(array):
    """
    Return `numpy` or `theano.tensor` depending on `array` type.

    Inspired by cupy.get_array_module:
    https://docs-cupy.chainer.org/en/latest/reference/generated/cupy.get_array_module.html

    """
    if isinstance(array, np.ndarray) or np.isscalar(array):
        return np
    else:
        return theano.tensor


def asarray(arrays):
    """
    Appropriately do `numpy.asarray` or `theano.tensor.as_tensor_variable`.
    """
    if is_theano(arrays):
        return arrays
    arrays = list(arrays)
    if any(map(is_theano, arrays)):
        return theano.tensor.as_tensor_variable(arrays)
    else:
        return np.asarray(arrays)


def gpu_context_to_dict(c):
    def jsonable(v):
        if isinstance(v, bytes):
            return v.decode()
        return v

    names = set(dir(c)) & {'bin_id', 'dev', 'devname', 'kind', 'pcibusid'}
    return {n: jsonable(getattr(c, n)) for n in names}


def theano_gpuarray_info():
    """
    Return information of GPUs used by Theano as a JSON'able dictionary.
    """
    return {
        k: gpu_context_to_dict(c)
        for k, c in theano.gpuarray.init_dev.devmap.items()
    }


def theano_info():
    return dict(
        gpuarray=theano_gpuarray_info(),
    )


def log_theano_info():
    gpu_info = theano_gpuarray_info()
    for key in sorted(gpu_info):
        logger.info('GPU (%s): %r', key, gpu_info[key])
