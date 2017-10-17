import inspect


class BaseComponent(object):

    @classmethod
    def consume_kwargs(cls, *args, **kwargs):
        sig = inspect.signature(cls)
        clskwds = {}
        for name, param in list(sig.parameters.items()):
            if param.kind in (param.POSITIONAL_OR_KEYWORD,
                              param.KEYWORD_ONLY):
                try:
                    clskwds[name] = kwargs.pop(name)
                except KeyError:
                    # Then let cls.__init__ bark, if the argument is
                    # required.
                    pass
        return cls(*args, **clskwds), kwargs

    @classmethod
    def from_dict(cls, dct):
        self, rest = cls.consume_kwargs(**dct)
        assert not rest
        return self
