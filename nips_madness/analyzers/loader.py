import collections
import json
import os

import numpy as np

try:
    string_types = (str, unicode)
except NameError:
    string_types = (str,)


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
    def load(cls, datastore):
        if os.path.isdir(datastore):
            dirname = datastore
        else:
            dirname = os.path.dirname(datastore)

        logpath = os.path.join(dirname, 'learning.csv')
        gen_logpath = os.path.join(dirname, 'generator.csv')
        tuning_logpath = os.path.join(dirname, 'TC_mean.csv')

        main_names, main = load_logfile(logpath)
        gen_names, gen = load_logfile(gen_logpath)
        tuning_names, tuning = load_logfile(tuning_logpath)

        with open(os.path.join(dirname, 'info.json')) as file:
            info = json.load(file)

        return cls(
            main_logpath=logpath,
            gen_logpath=gen_logpath,
            tuning_logpath=tuning_logpath,
            main=main, main_names=main_names,
            gen=gen, gen_names=gen_names,
            tuning=tuning, tuning_names=tuning_names,
            info=info)

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

        self.gen_index = self.gen[:, 0]
        self.log_J, self.log_D, self.log_S = \
            self.gen[:, 1:].reshape((-1, 3, 2, 2)).swapaxes(0, 1)

    def gen_param(self, name):
        return np.exp(getattr(self, 'log_' + name))

    def iter_gen_params(self, indices=None):
        if indices is None:
            indices = slice(None)
        for log_JDS in zip(self.log_J[indices],
                           self.log_D[indices],
                           self.log_S[indices]):
            yield list(map(np.exp, log_JDS))

    @property
    def track_offset_identity(self):
        try:
            return self.info['run_config']['track_offset_identity']
        except (AttributeError, KeyError):
            return False

    @property
    def sample_sites(self):
        try:
            return self.info['run_config']['sample_sites']
        except (AttributeError, KeyError):
            return 1

    @property
    def n_bandwidths(self):
        try:
            return len(self.info['run_config']['bandwidths'])
        except (AttributeError, KeyError):
            pass
        try:
            return self.info['run_config']['n_bandwidths']
        except (AttributeError, KeyError):
            return 8

    @property
    def n_bandwidths_viz(self):
        if self.track_offset_identity:
            return self.n_bandwidths * self.sample_sites
        else:
            return self.n_bandwidths

    @property
    def model_tuning(self):
        return self.tuning[:, :self.n_bandwidths_viz]

    @property
    def true_tuning(self):
        return self.tuning[:, self.n_bandwidths_viz:self.n_bandwidths_viz*2]

    @property
    def bandwidths(self):
        try:
            bandwidths = self.info['run_config']['bandwidths']
        except (AttributeError, KeyError):
            pass
        else:
            bandwidths = np.array(bandwidths)
            if self.track_offset_identity:
                # For visualization purpose, let's shift bandwidths of
                # different sample sites:
                return np.concatenate([
                    bandwidths + i for i in range(self.sample_sites)
                ])
            return bandwidths

        # Loading old logfiles:
        n = self.n_bandwidths
        if n == 8:
            return np.array([0, 0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1])
        elif n == 5:
            return np.array([0.0625, 0.125, 0.25, 0.5, 0.75])
        elif n == 4:
            return np.array([0.0625, 0.125, 0.25, 0.75])
        else:
            raise ValueError("Unknown n_bandwidth: {}".format(n))

    def to_dataframe(self):
        import pandas
        return pandas.DataFrame(self.main, columns=self.main_names)

    @property
    def gan_type(self):
        loss = self.info['run_config']['loss']
        try:
            return {
                'CE': 'RGAN',
                'LS': 'LSGAN',
                'WD': 'WGAN',
            }[loss]
        except KeyError:
            pass
        return '{}-GAN'.format(loss)

    def params(self):
        params = self.info['run_config']
        params = dict(params, gan_type=self.gan_type)
        return params

    def param_values(self, keys, default=None):
        params = self.params()
        return tuple(params.get(k, default) for k in keys)

    default_spec_keys = ('io_type', 'N', 'rate_cost', 'layers')

    def pretty_spec(self, keys=None):
        if keys is None:
            keys = self.default_spec_keys
        params = self.params()
        if 'layers' in params:
            params['layers'] = ','.join(map(str, params['layers']))
        keyvals = ' '.join('{}={}'.format(k, params[k]) for k in keys
                           if k in params)
        if params.get('N', None) == 0:
            for N in [51, 101, 102]:
                if params.get('datapath', '').endswith('Ne{}.mat'.format(N)):
                    params['N'] = N
                    break
        return '{}: {}'.format(params['gan_type'], keyvals)

    def __repr__(self):
        spec = self.pretty_spec()
        try:
            epochs = len(self.main)
        except AttributeError:
            epochs = 'UNKNOWN'
        return '<{} {spec} epochs={epochs}>'.format(
            self.__class__.__name__,
            **locals())


class GANGrid(object):

    def __init__(self, gans, sort_by='key_order'):
        gans = list(gans)
        self.diff = shallow_diff_dicts(data.params() for data in gans)
        if sort_by:
            if sort_by == 'key_order':
                sort_by = list(self.diff)
                for data in gans:
                    data.default_spec_keys = tuple(
                        k for k in sort_by
                        if k not in ('gan_type', 'WGAN'))
            gans.sort(key=lambda data: data.param_values(sort_by))
            self.diff = shallow_diff_dicts(data.params() for data in gans)
        self.gans = gans

    def to_dataframe(self):
        import pandas
        df = pandas.DataFrame(self.diff, columns=list(self.diff))
        df.loc[:, 'GANData'] = self.gans
        return df

    def __repr__(self):
        return '<{} axes={}>'.format(
            self.__class__.__name__,
            ','.join(map(str, self.diff)))


def shallow_diff_dicts(dicts):
    dicts = list(dicts)  # in case it is just an iterable
    keys = set()
    for d in dicts:
        keys |= set(d)
    keys = sorted(keys)

    def to_hashable(v):
        if isinstance(v, list):
            return tuple(v)
        return v

    missing = object()
    values = {k: [to_hashable(d.get(k, missing)) for d in dicts]
              for k in keys}
    hasdiff = {k: len(set(vs)) > 1 or missing in vs
               for k, vs in values.items()}
    return collections.OrderedDict((k, values[k]) for k in keys if hasdiff[k])


def load_gandata(paths, **kwds):
    if isinstance(paths, string_types):
        return GANData.load(paths)
    return GANGrid(map(GANData.load, paths), **kwds)
