import os

import numpy as np


def load_logfile(path):
    with open(path, 'rt') as file:
        first = file.readline()
        names = first.rstrip().split(',')
        try:
            float(names[0])
        except ValueError:
            skiprows = 1
        else:
            skiprows = 0
            names = []
    return names, np.loadtxt(path, delimiter=',', skiprows=skiprows)


def parse_tag(tag):
    if tag.startswith('data_'):
        use_data = True
    elif tag.startswith('generated_'):
        use_data = False
    else:
        return {}
    _, iot0, iot1, loss = tag.split('_', 3)
    io_type = '_'.join([iot0, iot1])
    if '_' in loss:
        loss, rest = loss.split('_', 1)
        layers = rest.split('_')
        del rest
    else:
        layers = []
    if layers and '.' in layers[-1]:
        rate_cost = float(layers[-1])
        layers = layers[:-1]
    else:
        rate_cost = 0
    layers = list(map(int, layers))
    del _, iot0, iot1, tag
    return locals()


class GANData(object):

    @classmethod
    def load(cls, logpath):
        basename = os.path.basename(logpath)
        dirname = os.path.dirname(logpath)
        suffix = basename[len('SSNGAN_log_'):]
        assert basename.startswith('SSNGAN_log_')
        tag = suffix[:-len('.log')]

        gen_logpath = os.path.join(dirname, 'parameters_' + suffix)
        tuning_logpath = os.path.join(dirname, 'D_parameters_' + suffix)

        main_names, main = load_logfile(logpath)
        gen_names, gen = load_logfile(gen_logpath)
        tuning_names, tuning = load_logfile(tuning_logpath)

        return cls(
            tag=tag,
            main_logpath=logpath,
            gen_logpath=gen_logpath,
            tuning_logpath=tuning_logpath,
            main=main, main_names=main_names,
            gen=gen, gen_names=gen_names,
            tuning=tuning, tuning_names=tuning_names,
            **parse_tag(tag))

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

        self.gen_index = self.gen[:, 0]
        self.log_J, self.log_D, self.log_S = \
            self.gen[:, 1:].reshape((-1, 3, 2, 2)).swapaxes(0, 1)

    def iter_gen_params(self, indices=None):
        if indices is None:
            indices = slice(None)
        for log_JDS in zip(self.log_J[indices],
                           self.log_D[indices],
                           self.log_S[indices]):
            yield list(map(np.exp, log_JDS))

    @property
    def model_tuning(self):
        return self.tuning[:, :8]

    @property
    def true_tuning(self):
        return self.tuning[:, 8:16]

    def to_dataframe(self):
        import pandas
        return pandas.DataFrame(self.main, columns=self.main_names)


load_gandata = GANData.load
