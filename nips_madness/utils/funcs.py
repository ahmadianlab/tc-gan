from collections import OrderedDict
import inspect

from .misc import add_arguments_from_dict


def default_arguments(func):
    return OrderedDict(
        (n, p.default) for n, p in inspect.signature(func).parameters.items()
        if (p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and
            p.default is not p.empty)
    )


def add_arguments_from_function(parser, func, help=None, **kwargs):
    if help is None:
        help = '{} parameters'.format(func.__name__)
    add_arguments_from_dict(parser, default_arguments(func),
                            help=help, **kwargs)
