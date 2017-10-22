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
    def consume_config(cls, config, *init_args, **init_kwargs):
        common = set(config) & set(init_kwargs)
        if common:
            raise ValueError(
                '{cls}.consume_config() got multiple values for'
                ' configurations: {}'.format(sorted(common), cls=cls))

        kwargs = dict(config, **init_kwargs)
        self, rest = cls.consume_kwargs(*init_args, **kwargs)

        unused = set(init_kwargs) - (set(init_kwargs) - set(rest))
        if unused:
            raise ValueError(
                '{cls}.consume_config() got following keyword arguments'
                ' which were NOT consumed by {cls}.consume_kwargs().'
                ' Make sure to use keyword arguments defined in'
                ' {cls}.consume_kwargs() and/or {cls}.__init__():'
                ' {}'.format(sorted(unused), cls=cls)
            )

        return self, rest

    @classmethod
    def from_dict(cls, dct):
        self, rest = cls.consume_kwargs(**dct)
        assert not rest
        return self


def consume_subdict(cls, key, dct, *args, **kwargs):
    rest = dict(dct)
    obj, subrest = cls.consume_config(rest.pop(key, {}), *args, **kwargs)
    if subrest:
        rest[key] = subrest
    return obj, rest
