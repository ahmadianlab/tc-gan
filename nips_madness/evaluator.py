import numpy as np


def eval_named(expr_ns, name, **kwargs):
    valuemap = {}
    for key, val in kwargs.items():
        sym = getattr(expr_ns, key)
        valuemap[sym] = np.asarray(val, dtype=sym.dtype)
    return getattr(expr_ns, name).eval(valuemap)


class MagicEvaluator(object):

    def __init__(self, expr_ns):
        self._expr_ns = expr_ns

    def __call__(self, name, **kwargs):
        return eval_named(self._expr_ns, name, **kwargs)

    def __getattr__(self, name):
        return lambda **kwargs: self(name, **kwargs)
