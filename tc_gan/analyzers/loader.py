"""
Old loader module.  # TODO: remove it
"""

from pathlib import Path
import collections
import json
import os
import warnings

import numpy as np

from .. import ssnode
from ..loaders.run_configs import guess_run_module

try:
    string_types = (str, unicode)
except NameError:
    string_types = (str,)


LogFile = collections.namedtuple('LogFile', ['names', 'data'])


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
    data = np.loadtxt(path, delimiter=',', skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape((1, -1))
    return LogFile(names, data)


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


class LazyLogFileLoader(object):

    def __init__(self, datastore, filenames):
        self.__datastore = Path(datastore)
        self.__filenames = filenames
        self.__namemap = {Path(name).stem: name for name in filenames}

    def __getattr__(self, name):
        if name not in self.__namemap:
            raise AttributeError('{} not in {}'.format(name, self.__namemap))
        try:
            return self.__dict__[name]
        except KeyError:
            pass
        path = self.__datastore.joinpath(self.__namemap[name])
        self.__dict__[name] = logfile = load_logfile(str(path))
        return logfile


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

        if gen.shape[-1] == 12:
            warnings.warn('generator.csv only contains 12 columns; '
                          'inserting index columns.')
            idx = np.arange(len(gen)).reshape((-1, 1))
            gen = np.concatenate([idx, gen], axis=1)

        if not main_names:
            warnings.warn('learning.csv has no header line; '
                          'assuming the default.')
            from ..run.gan import LearningRecorder
            assert main.shape[1] == len(LearningRecorder.column_names)
            main_names = LearningRecorder.column_names

        if len({len(main), len(gen), len(tuning)}) != 1:
            gen_step = min(len(main), len(gen), len(tuning))
            warnings.warn('Number of rows in learning.csv ({}),'
                          ' generator.csv ({}) and TC_mean.csv ({})'
                          ' are different.  Loading only ({}) rows.'
                          .format(len(main), len(gen), len(tuning),
                                  gen_step))
            main = main[:gen_step]
            gen = gen[:gen_step]
            tuning = tuning[:gen_step]

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
        if self.data_version < 1:
            self.log_J, self.log_D, self.log_S = self._gen_as_matrices()
            self.J = np.exp(self.log_J)
            self.D = np.exp(self.log_D)
            self.S = np.exp(self.log_S)
        else:
            self.J, self.D, self.S = self._gen_as_matrices()

    def _gen_as_matrices(self):
        return self.gen[:, 1:].reshape((-1, 3, 2, 2)).swapaxes(0, 1)

    @property
    def disc(self):
        if not hasattr(self, '_disc'):
            from .disc_learning import load_disc_log
            self._disc = load_disc_log(self.main_logpath)
        return self._disc

    def gen_param(self, name):
        return getattr(self, name)

    def true_param(self, name):
        return np.asarray(self.info['run_config']
                          .get('true_ssn_options', {})
                          .get(name, ssnode.DEFAULT_PARAMS[name]))

    def iter_gen_params(self, indices=None):
        if indices is None:
            indices = slice(None)
        return zip(self.J[indices], self.D[indices], self.S[indices])

    @property
    def data_version(self):
        try:
            return self.info['extra_info']['data_version']
        except (AttributeError, KeyError):
            return 0

    def fake_JDS(self):
        if self.data_version < 1:
            return np.exp(self.gen[:, 1:])
        else:
            return self.gen[:, 1:]

    def true_JDS(self):
        JDS = list(map(self.true_param, 'JDS'))
        return np.concatenate(JDS).flatten()

    @property
    def track_offset_identity(self):
        try:
            return self.info['run_config']['track_offset_identity']
        except (AttributeError, KeyError):
            if self.run_module == 'bptt_wgan':
                return True
            else:
                return False

    @property
    def include_inhibitory_neurons(self):
        try:
            return self.info['run_config']['include_inhibitory_neurons']
        except (AttributeError, KeyError):
            return False

    @property
    def n_sample_sites(self):
        try:
            sample_sites = self.info['run_config']['sample_sites']
        except (AttributeError, KeyError):
            return 1
        if self.include_inhibitory_neurons:
            return len(sample_sites) * 2
        else:
            return len(sample_sites)

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
            return self.n_stim * self.n_sample_sites
        else:
            return self.n_stim

    @property
    def contrasts(self):
        run_config = self.info['run_config']
        return run_config.get('contrasts') or run_config.get('contrast', [20])

    @property
    def n_contrasts(self):
        return len(self.contrasts)

    @property
    def n_stim(self):
        return self.n_bandwidths * self.n_contrasts

    @property
    def model_tuning(self):
        mid = self.tuning.shape[1] // 2
        return self.tuning[:, :mid]

    @property
    def true_tuning(self):
        mid = self.tuning.shape[1] // 2
        return self.tuning[:, mid:]

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
                    bandwidths + i for i in range(self.n_sample_sites)
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
        df = pandas.DataFrame(self.main, columns=self.main_names)
        if 'epoch' in df:
            df['gen_step'] = df['epoch']
            del df['epoch']  # otherwise pandas changes self.main inplace!
        df['epoch'] = self.gen_step_to_epoch(df['gen_step'])
        return df

    @property
    def total_steps(self):
        return len(self.main)

    @property
    def epochs(self):
        return self.gen_step_to_epoch(self.main[:, 0])

    @property
    def datasize(self):
        run_config = self.info['run_config']
        if self.run_module == 'bptt_cwgan':
            truth_size = run_config['truth_size']
            try:
                num_probes = len(run_config['norm_probes'])
            except KeyError:
                num_probes = len(run_config['probe_offsets'])
            num_contrasts = len(run_config['contrasts'])
            if run_config['include_inhibitory_neurons']:
                coeff = (0.8 + 0.2) / (0.8 + 0.8)
                return truth_size * num_probes * num_contrasts * coeff
            else:
                return truth_size * num_probes * num_contrasts
        else:
            return run_config['truth_size']

    def gen_step_to_epoch(self, gen_step):
        disc_updates = self.gen_step_to_disc_updates(gen_step)
        return disc_updates * self.batchsize / self.datasize

    def gen_step_to_disc_updates(self, gen_step):
        if self.is_WGAN:
            WGAN_n_critic0 = self.critic_iters_init
            WGAN_n_critic = self.critic_iters
            return WGAN_n_critic0 + WGAN_n_critic * gen_step
        else:
            return gen_step + 1

    def epoch_to_gen_step(self, epoch):
        return self.disc_updates_to_gen_step(self.epoch_to_disc_updates(epoch))

    def disc_updates_to_gen_step(self, disc_updates):
        if self.is_WGAN:
            WGAN_n_critic0 = self.critic_iters_init
            WGAN_n_critic = self.critic_iters
            return (disc_updates - WGAN_n_critic0) / WGAN_n_critic
        else:
            return disc_updates - 1

    def epoch_to_disc_updates(self, epoch):
        return epoch * self.datasize / self.batchsize

    @property
    def batchsize(self):
        run_config = self.info['run_config']
        try:
            return run_config['num_models'] * run_config['probes_per_model']
        except KeyError:
            pass
        try:
            return run_config['batchsize']
        except KeyError:
            return run_config['n_samples']

    @property
    def critic_iters(self):
        run_config = self.info['run_config']
        try:
            return run_config['critic_iters']
        except KeyError:
            return run_config.get('WGAN_n_critic', 5)

    @property
    def critic_iters_init(self):
        run_config = self.info['run_config']
        try:
            return run_config['critic_iters_init']
        except KeyError:
            return run_config.get('WGAN_n_critic0', 50)

    @property
    def is_WGAN(self):
        return self.gan_type in ('cWGAN', 'WGAN')

    @property
    def gan_type(self):
        try:
            loss = self.info['run_config']['loss']
        except KeyError:
            if self.run_module == 'bptt_cwgan':
                return 'cWGAN'
            elif self.run_module == 'bptt_wgan':
                loss = 'WD'
            else:
                raise
        try:
            return {
                'CE': 'RGAN',
                'LS': 'LSGAN',
                'WD': 'WGAN',
            }[loss]
        except KeyError:
            pass
        return '{}-GAN'.format(loss)

    @property
    def run_module(self):
        return guess_run_module(self.info)

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


def get_run_module(path):
    datastore = Path(path)
    if not datastore.is_dir():
        datastore = datastore.parent
    with open(str(datastore.joinpath('info.json'))) as file:
        info = json.load(file)
    return guess_run_module(info)


def get_loader(paths):
    if isinstance(paths, string_types):
        first_path = paths
    else:
        first_path = paths[0]
    run_module = get_run_module(first_path)
    if run_module == 'bptt_moments':
        from .mm_loader import MomentMatchingData
        if isinstance(paths, string_types):
            return MomentMatchingData
        else:
            return lambda paths: map(MomentMatchingData, paths)
    else:
        return load_gandata


def load_learning(paths, **kwds):
    return get_loader(paths)(paths, **kwds)
