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


def consume_config(emitter, config, *args, **kwargs):
    """
    Call `emitter` using a subset of `config` and return the rest.

    It is conceptually equivalent to::

      cls.consume_kwargs(*args, **dict(config, **kwargs))

    However, it does following extra checks:

    * Make sure `config` and `kwargs` do NOT contain any shared keys.
    * Make sure `kwargs` is used by `emitter`.

    """
    common = set(config) & set(kwargs)
    if common:
        raise ValueError(
            'Trying to pass multiple values for configurations to'
            ' {emitter}(): {}'.format(sorted(common), emitter=emitter))

    total_kwargs = dict(config, **kwargs)
    obj, rest = emitter(*args, **total_kwargs)

    unused = set(kwargs) - (set(kwargs) - set(rest))
    if unused:
        raise ValueError(
            'consume_config({emitter}, ...) got the following keyword'
            ' arguments which were NOT consumed by {emitter}().  Make'
            ' sure to pass keyword arguments consumed by {emitter}(): {}'
            .format(sorted(unused), emitter=emitter)
        )

    return obj, rest


def consume_subdict(emitter, key, dct, *args, **kwargs):
    """
    Return ``emitter(*args, **dct[key], **kwargs)`` with unused part of `dct`.

    The sub-dictionary ``dct[key]`` may not exist (assumed be an empty
    dictionary if so) and it is removed if all contents in
    ``dct[key]`` are consumed.  Thus, it is safe to call this function
    with the same `key` and different `emitter` multiple times.

    """
    rest = dict(dct)
    obj, subrest = consume_config(emitter, rest.pop(key, {}), *args, **kwargs)
    if subrest:
        rest[key] = subrest
    return obj, rest
