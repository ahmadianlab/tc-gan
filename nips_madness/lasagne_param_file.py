import lasagne
import numpy as np


def dump(layer, path):
    np.savez_compressed(
        path,
        **{
            str(i): p for i, p in
            enumerate(lasagne.layers.get_all_param_values(layer))
        })


def load(path):
    npz = np.load(path)
    _keys, values = zip(*sorted(npz.items(), key=lambda kv: int(kv[0])))
    return list(values)
