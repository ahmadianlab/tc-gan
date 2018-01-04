from collections import OrderedDict
import inspect


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
