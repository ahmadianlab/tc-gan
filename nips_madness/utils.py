from __future__ import print_function

from contextlib import contextmanager
from logging import getLogger
import multiprocessing
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import theano

logger = getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_meta_info(packages=[]):
    return dict(
        repository=dict(
            revision=git_revision(),
            is_clean=git_is_clean(),
        ),
        python=sys.executable,
        packages={p.__name__: p.__version__ for p in packages},
        argv=sys.argv,
    )


def git_revision():
    return git_output(['git', 'rev-parse', 'HEAD']).rstrip()


def git_is_clean():
    return git_output(['git', 'status', '--short',
                       '--untracked-files=no']).strip() == ''


def git_output(args):
    return subprocess.check_output(
        args,
        cwd=PROJECT_ROOT,
        universal_newlines=True)


def make_progressbar(quiet=False, **kwds):
    if quiet:
        return lambda xs: xs
    else:
        try:
            import progressbar
        except ImportError:
            def dummy_bar(xs):
                for x in xs:
                    print('.', end='')
                    yield x
                print()
            return dummy_bar
        else:
            return progressbar.ProgressBar(**kwds)


class StopWatch(object):

    """
    Context manager based stop watch:

    >>> sw = StopWatch()
    >>> for _ in range(10):
    ...     with sw:
    ...         time.sleep(0.1)
    >>> print('{:.1f}'.format(sw.sum()))
    1.0

    """

    def __init__(self):
        self.times = []

    def __enter__(self):
        self.pre = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.times.append(time.time() - self.pre)

    def sum(self):
        return sum(self.times)


@contextmanager
def log_timing(opname='Operation', logger=logger):
    pre = time.time()
    yield
    t = time.time() - pre
    logger.info('%s took %s seconds', opname, t)


def csv_line(value_parser):
    """
    Return a function that parses a line of comma separated values.

    Examples:

    >>> csv_line(int)('1, 2, 3')
    [1, 2, 3]
    >>> csv_line(float)('0.5, 1.5')
    [0.5, 1.5]

    For example, it can be passed to type argument of
    `argparse.ArgumentParser.add_argument` as follows::

        parser.add_argument(
            ...,
            type=csv_line(float),
            help='Comma separated value of floats')

    """
    def convert(string):
        return list(map(value_parser, string.split(',')))
    return convert


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


def cpu_count(_environ=os.environ):
    """ Return available number of CPUs; Slurm/PBS-aware version. """
    try:
        return int(_environ['SLURM_CPUS_ON_NODE'])
    except (KeyError, ValueError):
        pass
    try:
        return int(_environ['PBS_NUM_PPN'])
    except (KeyError, ValueError):
        pass
    return multiprocessing.cpu_count()


class cached_property(object):
    """
    A decorator that converts a function into a lazy property.

    The function wrapped is called the first time to retrieve the
    result and then that calculated result is used the next time you
    access the value::

        class Foo(object):

            @cached_property
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to
    work.

    This class is stolen from `werkzeug.utils`_ with a little bit of
    modification.  See also `Python cached property decorator`_.

    * `werkzeug.utils
      <https://github.com/mitsuhiko/werkzeug/blob/master/werkzeug/utils.py>`_
    * toofishes.net - `Python cached property decorator
      <http://www.toofishes.net/blog/python-cached-property-decorator/>`_

    Examples:

    >>> class Foo(object):
    ...
    ...     counter = 0
    ...
    ...     @cached_property
    ...     def foo(self):
    ...         self.counter += 1
    ...         return self.counter
    ...
    >>> ya = Foo()
    >>> ya.foo
    1

    The result is cached.  So you should get the same result as before.

    >>> ya.foo
    1

    You can invalidate the cache by ``del obj.property``:

    >>> del ya.foo
    >>> ya.foo
    2

    """

    # implementation detail: this property is implemented as non-data
    # descriptor.  non-data descriptors are only invoked if there is
    # no entry with the same name in the instance's __dict__.
    # this allows us to completely get rid of the access function call
    # overhead.  If one choses to invoke __get__ by hand the property
    # will still work as expected because the lookup logic is replicated
    # in __get__ for manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func
        self._missing = object()

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


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

    rng = np.random.RandomState(seed)

    def iterator():
        while True:
            idx = np.arange(len(data))
            rng.shuffle(idx)

            for i in range(num_batches):
                s = i * batchsize
                e = (i + 1) * batchsize
                yield data[idx[s:e]]

    return iterator()


def theano_function(*args, **kwds):
    # from theano.compile.nanguardmode import NanGuardMode
    # kwds.setdefault('mode', NanGuardMode(
    #     nan_is_error=True, inf_is_error=True, big_is_error=True))
    kwds.setdefault('allow_input_downcast', True)
    # MAYBE: make sure to use theano.config.floatX everywhere and
    # remove allow_input_downcast.
    return theano.function(*args, **kwds)


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


def param_module(path):
    if path.lower().endswith(('.yaml', '.yml')):
        import yaml
        return yaml, ''
    elif path.lower().endswith('.json'):
        import json
        return json, ''
    elif path.lower().endswith(('.pickle', '.pkl')):
        try:
            import cPickle as pickle
        except:
            import pickle
        return pickle, 'b'
    elif path.lower().endswith('.toml'):
        import toml
        return toml, ''
    else:
        raise ValueError(
            'data format of {!r} is not supported'.format(path))


def load_any_file(path):
    """
    Load data from given path; data format is determined by file extension
    """
    module, mode = param_module(path)
    with open(path, 'r' + mode) as f:
        return module.load(f)


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


def objectpath(obj):
    """
    Get an importable path of an object `obj`.

    >>> import json
    >>> objectpath(json.load)
    'json.load'
    """
    return obj.__module__ + '.' + obj.__name__
