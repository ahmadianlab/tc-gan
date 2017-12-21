"""
Functions to generate tuning curve dataset ("true data").
"""

from logging import getLogger

import numpy as np

from .. import ssnode
from .. import utils
from .fixed_time_sampler import FixedTimeTuningCurveSampler
from .wgan import is_heteroin

logger = getLogger(__name__)


def dataset_by_ssnode(
        num_sites, bandwidths, contrasts,
        truth_size, truth_seed,
        sample_sites, include_inhibitory_neurons,
        true_ssn_options={}):

    with utils.log_timing('sample_tuning_curves()'):
        data, (_, _, fpinfo) = ssnode.sample_tuning_curves(
            sample_sites=sample_sites,
            NZ=truth_size,
            seed=truth_seed,
            bandwidths=bandwidths,
            contrast=contrasts,
            N=num_sites,
            track_offset_identity=True,
            include_inhibitory_neurons=include_inhibitory_neurons,
            **dict(dict(
                dt=5e-4,  # as in run/gan.py
                max_iter=100000,
                io_type='asym_power',
                rate_stop_at=200,
                # For io_type='asym_power' (and 'asym_linear'),
                # `rate_stop_at` determines the rate at which the
                # solver terminates and reject the sample.  Let's use
                # this feature to exclude extremely large rate in the
                # tuning curve data.  But let's not make it too large,
                # since transiently having large rate is OK.
                #
                # TODO: Add another rate limiting which acts only on
                #       the tuning curve data (i.e., sub-sampled fixed
                #       point) to allow large transient rate and large
                #       rates in non-sampled locations.
            ), **true_ssn_options))
    data = np.array(data.T)      # shape: (N_data, nb)

    logger.info('Information of sampled tuning curves:')
    logger.info('  len(data): %s', len(data))
    logger.info('  rejections: %s', fpinfo.rejections)
    logger.info('  rejection rate'
                ' = rejections / (rejections + len(data)) : %s',
                fpinfo.rejections / (fpinfo.rejections + len(data)))
    logger.info('  error codes: %r', fpinfo.counter)

    return data


def dataset_by_fixedtime(
        learner, truth_size, truth_seed,
        truth_batchsize=50, true_ssn_options={}):

    if truth_size < truth_batchsize:
        repeat = 1
        truth_batchsize = truth_size
    else:
        repeat, mod = divmod(truth_size, truth_batchsize)
        if mod:
            repeat += 1
            logger.warn(
                'truth_size=%d is not divisible by truth_batchsize=%d.'
                ' You are wasting %d samples.',
                truth_size, truth_batchsize,
                truth_batchsize * repeat - truth_size)

    # Collect learnable parameters:
    learnable_parameters = {}
    sampler_options = dict(true_ssn_options)
    for param in learner.gen.get_all_params():
        try:
            learnable_parameters[param.name] = sampler_options.pop(param.name)
        except KeyError:
            continue

    sampler = FixedTimeTuningCurveSampler.from_learner(
        learner,
        batchsize=truth_batchsize,
        seed=truth_seed,
        **sampler_options)  # let sampler bark if it has unsupported options

    sampler.gen.set_params(learnable_parameters)
    sampler.prepare()

    data_list = []
    dynamics_penalty_list = []
    with utils.log_timing('sampler.forward() x{}'.format(repeat)):
        for i in range(repeat):
            with utils.log_timing('sampler.forward() ({}/{})'
                                  .format(i + 1, repeat)):
                out = sampler.forward(full_output=True)
            dynamics_penalty_list.append(out.model_dynamics_penalty)
            data_list.append(out.prober_tuning_curve)

    dynamics_penalty = np.array(dynamics_penalty_list)
    logger.info('Statistics of dynamics_penalty:')
    logger.info('  mean: %s', dynamics_penalty.mean())
    logger.info('  std : %s', dynamics_penalty.std())
    logger.info('  min : %s', dynamics_penalty.min())
    logger.info('  25%% : %s', np.percentile(dynamics_penalty, 25))
    logger.info('  50%% : %s', np.percentile(dynamics_penalty, 50))
    logger.info('  75%% : %s', np.percentile(dynamics_penalty, 75))
    logger.info('  max : %s', dynamics_penalty.max())

    data = np.concatenate(data_list)
    return data[:truth_size]


dataset_provider_choices = ('ssnode', 'fixedtime')
"""
Possible methods for generating dataset.

See: `generate_dataset`.
"""


def generate_dataset(learner,
                     dataset_provider='ssnode',
                     **kwargs):
    """
    Generate "truth" dataset for a given `learner`.

    Parameters
    ----------
    learner : `.BPTTWassersteinGAN`, `.BPTTMomentMatcher`, etc.
        A GAN or moment-matcher.

    dataset_provider : {'ssnode', 'fixedtime'}
        Method for generating dataset:

        ssnode
            Use `.ssnode.sample_tuning_curves` to generate dataset.  It
            defaults to parallelized fast C implementation of fixed point
            finder.

        fixedtime
            Use `.FixedTimeTuningCurveSampler` which, in turn, uses
            `.TuningCurveGenerator.forward`.  It is the same method as
            the one used during learning.  However, the convergence of
            the solutions are not verified.

    truth_size : int

    truth_seed : int

    true_ssn_options : dict

    """
    if dataset_provider == 'ssnode' and is_heteroin(learner.gen):
        # TODO: support heteroin in ssnode
        raise NotImplementedError("ssnode does not support SSN with"
                                  " heterogeneous input (yet).")

    logger.info('Generating the truth...')
    if dataset_provider == 'ssnode':
        return dataset_by_ssnode(
            num_sites=learner.gen.num_sites,
            bandwidths=learner.bandwidths,
            contrasts=learner.contrasts,
            sample_sites=learner.sample_sites,
            include_inhibitory_neurons=learner.include_inhibitory_neurons,
            **kwargs)
    elif dataset_provider == 'fixedtime':
        return dataset_by_fixedtime(
            learner,
            **kwargs)
    else:
        raise ValueError('Unknown dataset_provider: {}'
                         .format(dataset_provider))
