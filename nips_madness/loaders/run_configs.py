from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..utils import iteritemsdeep, getdeep


def parse_gen_param_name(name):
    """
    Convert ``'J'`` to ``('J', None)`` and ``'J_EI'`` to ``('J', (0, 1))``.

    >>> parse_gen_param_name('J')
    ('J', None)
    >>> parse_gen_param_name('D_IE')
    ('D', (1, 0))
    >>> parse_gen_param_name('V_E')
    ('V', (0,))
    >>> parse_gen_param_name('A_xx')
    Traceback (most recent call last):
      ...
    ValueError: Unsupported suffix in A_xx

    """
    suffix_to_index = {'E': 0, 'I': 1}
    if '_' in name:
        array_name, suffix = name.split('_', 1)
        if set(suffix) - set(suffix_to_index):
            raise ValueError('Unsupported suffix in {}'.format(name))
        index = tuple(suffix_to_index[i] for i in suffix)
        return array_name, index
    else:
        return name, None


def guess_run_module(info):
    try:
        script_file = info['extra_info']['script_file']
    except KeyError:
        return 'gan'
    return Path(script_file).stem


class BaseRunConfig(Mapping):

    is_legacy = False

    @classmethod
    def from_info(cls, info):
        return cls(info['run_config'])

    def __init__(self, run_config):
        self.dict = run_config

    def __getattr__(self, name):
        try:
            return self.dict[name]
        except KeyError:
            raise AttributeError(name)

    def items(self):
        return (('.'.join(key), val) for key, val in iteritemsdeep(self.dict))

    def __iter__(self):
        return (key for key, _ in self.items())

    def __len__(self):
        return sum(1 for _ in iter(self))

    def __getitem__(self, key):
        return getdeep(self.dict, key)

    @property
    def datasize(self):
        return self.dict['truth_size']

    @property
    def n_bandwidths(self):
        return len(self.bandwidths)

    @property
    def n_contrasts(self):
        return len(self.contrasts)

    @property
    def cell_types(self):
        return [0, 1] if self.include_inhibitory_neurons else [0]

    @property
    def norm_probes(self):
        try:
            return self.dict['norm_probes']
        except KeyError:
            pass
        try:
            return self.dict['sample_sites']
        except KeyError:
            pass
        raise AttributeError("{} has no attribute 'norm_probes'".format(self))

    @property
    def ssn_impl(self):
        return self.dict.get('ssn_impl', 'default')

    @property
    def ssn_type(self):
        return self.dict.get('ssn_type', 'default')

    # @property
    # def n_stim(self):
    #     return self.n_bandwidths * self.n_contrasts

    def get_true_param(self, name):
        from .. import ssnode
        true_ssn_options = self.dict.get('true_ssn_options')
        array_name, index = parse_gen_param_name(name)
        if index is None:
            if name in true_ssn_options:
                val = true_ssn_options[name]
            elif name in ssnode.DEFAULT_PARAMS:
                val = ssnode.DEFAULT_PARAMS[name]
            else:
                raise ValueError('Unknown parameter: {}'.format(name))
            return np.asarray(val)
        else:
            return self.get_true_param(array_name)[index]


class BaseGANRunConfig(BaseRunConfig):

    def gen_step_to_epoch(self, gen_step):
        disc_updates = self.gen_step_to_disc_updates(gen_step)
        return self.disc_updates_to_epoch(disc_updates)

    def disc_updates_to_epoch(self, disc_updates):
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


class LegacyGANRunConfig(BaseGANRunConfig):

    is_legacy = True

    @property
    def track_offset_identity(self):
        return self.dict.get('track_offset_identity', False)

    @property
    def include_inhibitory_neurons(self):
        return self.dict.get('include_inhibitory_neurons', False)

    @property
    def n_sample_sites(self):
        try:
            sample_sites = self.dict['sample_sites']
        except KeyError:
            return 1
        if self.include_inhibitory_neurons:
            return len(sample_sites) * 2
        else:
            return len(sample_sites)

    @property
    def n_bandwidths(self):
        try:
            return len(self.dict['bandwidths'])
        except KeyError:
            pass
        try:
            return self.dict['n_bandwidths']
        except KeyError:
            return 8

    @property
    def contrasts(self):
        return self.dict.get('contrast', [20])

    @property
    def contrast(self):
        raise AttributeError("Accessing `contrast` not allowed."
                             " Use `contrasts`.")

    @property
    def bandwidths(self):
        try:
            return self.dict['bandwidths']
        except KeyError:
            pass

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

    @property
    def batchsize(self):
        return self.dict['n_samples']

    @property
    def critic_iters(self):
        return self.dict.get('WGAN_n_critic', 5)

    @property
    def critic_iters_init(self):
        return self.dict.get('WGAN_n_critic0', 50)

    @property
    def gan_type(self):
        loss = self.dict['loss']
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
    def is_WGAN(self):
        return self.gan_type in ('cWGAN', 'WGAN')


class BaseWGANRunConfig(BaseGANRunConfig):
    track_offset_identity = True
    is_WGAN = True


class BPTTWGANRunConfig(BaseWGANRunConfig):
    gan_type = 'WGAN'


class BPTTcWGANRunConfig(BaseWGANRunConfig):
    gan_type = 'cWGAN'

    @property
    def batchsize(self):
        run_config = self.dict
        return run_config['num_models'] * run_config['probes_per_model']

    @property
    def datasize(self):
        run_config = self.dict
        truth_size = run_config['truth_size']
        try:
            num_probes = len(run_config['norm_probes'])
        except KeyError:
            num_probes = len(run_config['probe_offsets'])
        num_contrasts = len(run_config['contrasts'])
        if run_config['include_inhibitory_neurons']:
            # coeff = (0.8 + 0.2) / (0.8 + 0.8)
            coeff = 1 / (2 * self.e_ratio)
            return truth_size * num_probes * num_contrasts * coeff
        else:
            return truth_size * num_probes * num_contrasts

    @property
    def e_ratio(self):
        return self.dict.get('e_ratio', 0.8)


class BPTTMomentsRunConfig(BaseRunConfig):

    @property
    def num_mom_conds(self):
        """
        Number of conditions in which moments are evaluated.
        """
        return (self.n_bandwidths * self.n_contrasts *
                len(self.cell_types) * len(self.norm_probes))
    # See also: [[../networks/moment_matching.py::num_mom_conds]]

    def step_to_epoch(self, step):
        return step * self.batchsize / self.datasize

    def epoch_to_step(self, epoch):
        return epoch / self.batchsize * self.datasize


module_class_map = {
    'bptt_wgan': BPTTWGANRunConfig,
    'bptt_cwgan': BPTTcWGANRunConfig,
    'bptt_moments': BPTTMomentsRunConfig,
    'gan': LegacyGANRunConfig,
    'cgan': LegacyGANRunConfig,
}


def get_run_config(info):
    run_module = guess_run_module(info)
    return module_class_map[run_module].from_info(info)
