import inspect


class BaseComponent(object):

    """
    A base class for composable components.

    This class helps propagating, sharing, and intercepting arguments
    (initialization parameters) in a unified manner so that classes
    can pass arguments to their sub-components reliably.

    For a basic usage, `BaseComponent` only requires its subclass to
    *not* have keyword arguments (``**kwargs``) for its `__init__`
    method.  Example:

    >>> class Wheel(BaseComponent):
    ...     def __init__(self, spokes):
    ...         self.spokes = spokes
    ...

    Then, `BaseComponent` gives extra ways to instantiate the class,
    such as `.consume_kwargs`:

    >>> wheel, rest = Wheel.consume_kwargs(spokes=28, something_else=0)
    >>> wheel.spokes
    28
    >>> rest
    {'something_else': 0}

    `.consume_kwargs` is simple but it provides a powerful way of
    composing multiple objects together, by "chaining" them:

    >>> class Handle(BaseComponent):
    ...     def __init__(self, brakes):
    ...         self.brakes = brakes
    ...
    >>> class Bike(BaseComponent):
    ...
    ...     @classmethod
    ...     def consume_kwargs(cls, **kwargs):
    ...         wheel, rest = Wheel.consume_kwargs(**kwargs)
    ...         handle, rest = Handle.consume_kwargs(**rest)
    ...         return super(Bike, cls).consume_kwargs(wheel, handle, **rest)
    ...
    ...     def __init__(self, wheel, handle, color):
    ...         self.wheel = wheel
    ...         self.handle = handle
    ...         self.color = color
    ...
    >>> bike, rest = Bike.consume_kwargs(spokes=28, brakes=2, color='brown')
    >>> rest
    {}
    >>> bike.wheel.spokes
    28
    >>> bike.handle.brakes
    2
    >>> bike.color
    'brown'

    Such chaining, by default, makes sure that every keyword argument
    given is used at most once.  To make sure that all keyword
    arguments are consumed, use `.from_dict`:

    >>> _ = Bike.from_dict(dict(spokes=28, brakes=2, color='brown'))
    >>> Bike.from_dict(dict(spokes=28, brakes=2, color='brown', spam=1))
    Traceback (most recent call last):
      ...
    ValueError: Not all key-value pairs are consumed: ['spam']

    """

    @classmethod
    def consume_kwargs(cls, *args, **kwargs):
        """
        Instantiate `cls` using a subset of `kwargs` and return the rest.

        Returns
        -------
        self : cls
            The result of ``cls(*args, **clskwds)`` where `clskwds` is
            a subset of `kwargs`.  It is determined by the call
            signature of `cls.__init__`.
        rest : dict
            A subset of `kwargs` not used by `cls.__init__`; i.e.,
            ``rest = kwargs - clskwds``.

        """
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
        """
        Instantiate `cls` using a subset of `config` and return the rest.

        It is conceptually equivalent to::

          cls.consume_kwargs(*init_kwargs, **dict(config, **init_kwargs))

        However, it does following extra checks:

        * Make sure `config` and `init_kwargs` do NOT contain any
          shared keys.
        * Make sure `init_kwargs` is used by `cls.__init__`.

        """
        # MAYBE: Turn consume_config into a function?  Not sure if
        # subclasses want to override this method.
        common = set(config) & set(init_kwargs)
        if common:
            raise ValueError(
                '{cls}.consume_config() got multiple values for'
                ' configurations: {}'.format(sorted(common),
                                             cls=cls.__name__))

        kwargs = dict(config, **init_kwargs)
        self, rest = cls.consume_kwargs(*init_args, **kwargs)

        unused = set(init_kwargs) - (set(init_kwargs) - set(rest))
        if unused:
            raise ValueError(
                '{cls}.consume_config() got following keyword arguments'
                ' which were NOT consumed by {cls}.consume_kwargs().'
                ' Make sure to use keyword arguments defined in'
                ' {cls}.consume_kwargs() and/or {cls}.__init__():'
                ' {}'.format(sorted(unused),
                             cls=cls.__name__)
            )

        return self, rest

    @classmethod
    def from_dict(cls, dct):
        """
        Instantiate `cls` using a dictionary `dct`.

        It is equivalent to ``cls.consume_kwargs(**dct)`` but makes
        sure all keys are used.

        """
        self, rest = cls.consume_kwargs(**dct)
        if rest:
            raise ValueError(
                'Not all key-value pairs are consumed: {}'
                .format(sorted(rest)))
        return self


def consume_subdict(cls, key, dct, *args, **kwargs):
    """
    Instantiate `cls` using ``dct[key]`` and return the rest.

    The sub-dictionary ``dct[key]`` may not exist (assumed be an empty
    dictionary if so) and it is removed if all contents in
    ``dct[key]`` are consumed.  Thus, it is safe to call this function
    with the same `key` and different `cls` multiple times.

    """
    rest = dict(dct)
    obj, subrest = cls.consume_config(rest.pop(key, {}), *args, **kwargs)
    if subrest:
        rest[key] = subrest
    return obj, rest
