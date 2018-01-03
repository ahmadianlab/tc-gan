from __future__ import print_function

from collections import OrderedDict
from contextlib import contextmanager
from logging import getLogger
import inspect
import time

logger = getLogger(__name__)


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


def objectpath(obj):
    """
    Get an importable path of an object `obj`.

    >>> import json
    >>> objectpath(json.load)
    'json.load'
    """
    return obj.__module__ + '.' + obj.__name__


def default_arguments(func):
    return OrderedDict(
        (n, p.default) for n, p in inspect.signature(func).parameters.items()
        if (p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and
            p.default is not p.empty)
    )


def add_arguments_from_function(parser, func, help=None, exclude=[], **kwargs):
    if help is None:
        help = '{} parameters'.format(func.__name__)
    for key, val in default_arguments(func).items():
        if key in exclude:
            continue
        if isinstance(val, (str, float, int)):
            argtype = type(val)
        else:
            argtype = eval
        parser.add_argument(
            '--{}'.format(key.replace('_', '-')),
            type=argtype, default=val, help=help,
            **kwargs)


class Namespace(object):

    """
    Like `types.SimpleNamespace` but less verbose.

    >>> ns = Namespace(a=1, b=2, c=3)
    >>> ns
    <Namespace[3]: a b c>
    >>> ns.a
    1
    >>> Namespace(**{'a{:02}'.format(i): 0 for i in range(20)})
    <Namespace[20]: a00 a01 a02 a03 a04 a05 a06 a07 a08 a09 ...>

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        keys = sorted(self.__dict__)
        limit = 10
        if len(keys) > limit:
            keys = keys[:limit] + ['...']

        return '<{}[{}]: {}>'.format(
            type(self).__name__,
            len(self.__dict__),
            ' '.join(keys))
