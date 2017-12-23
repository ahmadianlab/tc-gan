import abc

import numpy as np

from ..networks.ssn import concat_flat


class AbstractGANRunConfig(abc.ABC):

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
    def ssn_impl(self):
        return self.dict.get('ssn_impl', 'default')

    @property
    def ssn_type(self):
        return self.dict.get('ssn_type', 'default')

    @property
    def param_array_names(self):
        # TODO: save param_array_names in info.json
        if self.ssn_type == 'heteroin':
            # See: AbstractEulerSSNCore.get_flat_param_names
            return ['V', 'J', 'D', 'S']
        else:
            return ['J', 'D', 'S']

    @property
    def param_element_names(self):
        # TODO: save param_element_names in info.json
        from ..networks.ssn import genparam_names
        names = list(genparam_names)
        if self.ssn_type == 'heteroin':
            names = ['V_E', 'V_I'] + names
        return names

    # @property
    # def n_stim(self):
    #     return self.n_bandwidths * self.n_contrasts

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

    def get_true_param(self, name):
        from .. import ssnode
        true_ssn_options = self.dict.get('true_ssn_options')
        if name in true_ssn_options:
            val = true_ssn_options[name]
        elif name in ssnode.DEFAULT_PARAMS:
            val = ssnode.DEFAULT_PARAMS[name]
        return np.asarray(val)

    def flatten_true_params(self, ndim=1):
        assert ndim in (1, 2)
        params = concat_flat(map(self.get_true_param,
                                 self.param_array_names))
        if ndim == 2:
            return params.reshape((1, -1))
        return params


class LegacyGANRunConfig(AbstractGANRunConfig):

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


class AbstractWGANRunConfig(AbstractGANRunConfig):
    track_offset_identity = True
    is_WGAN = True


class BPTTWGANRunConfig(AbstractWGANRunConfig):
    gan_type = 'WGAN'


class BPTTcWGANRunConfig(AbstractWGANRunConfig):
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


module_class_map = {
    'bptt_wgan': BPTTWGANRunConfig,
    'bptt_cwgan': BPTTcWGANRunConfig,
}


def get_run_config(info):
    from ..analyzers.loader import guess_run_module
    run_module = guess_run_module(info)
    return module_class_map.get(
        run_module, LegacyGANRunConfig,
    ).from_info(info)
