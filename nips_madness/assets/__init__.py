import os


ROOT = os.path.dirname(os.path.realpath(__file__))


def path(*args):
    """ File path to asset file. """
    return os.path.join(ROOT, *args)
