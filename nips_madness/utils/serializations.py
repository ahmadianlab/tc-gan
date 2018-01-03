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
