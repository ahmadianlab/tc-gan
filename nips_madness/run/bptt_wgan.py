"""
Run SSN-BTTP GAN learning.
"""

from logging import getLogger

import numpy as np

from . import gan as plain_gan
from .. import ssnode
from .. import utils
from ..drivers import GANDriver
from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..networks.bptt_gan import make_gan
from ..recorders import UpdateResult

logger = getLogger(__name__)


def learn(
        driver, truth_size, truth_seed,
        sample_sites, include_inhibitory_neurons,
        ):

    np.random.seed(0)
    gan = driver.gan
    logger.info('Compiling Theano functions...')
    with utils.log_timing('gan.prepare()'):
        gan.prepare()
    ssn = gan.gen
    sample_sites = sample_sites_from_stim_space(sample_sites, ssn.num_sites)

    logger.info('Generating the truth...')
    with utils.log_timing('sample_tuning_curves()'):
        data, _ = ssnode.sample_tuning_curves(
            sample_sites=sample_sites,
            NZ=truth_size,
            seed=truth_seed,
            bandwidths=gan.bandwidths,
            contrast=gan.contrasts,
            N=ssn.num_sites,
            track_offset_identity=True,
            include_inhibitory_neurons=include_inhibitory_neurons,
            # dt=ssn.dt,
            dt=5e-4,  # as in ./gan.py
            max_iter=100000,
            r0=np.zeros(2 * ssn.num_sites),
            io_type='asym_power',
        )
    data = np.array(data.T)      # shape: (N_data, nb)

    with utils.log_timing('numpy.save("truth.npy", data)'):
        np.save(driver.datastore.path('truth.npy'), data)
    # Note: saving in .npy rather than .npz so that it can be
    # deduplicated (by, e.g., git-annex).

    gan.init_dataset(data)

    learning_it = gan.learning()

    @driver.iterate
    def _(k):
        # This callback function "connects" gan.learning and drive.iterate.
        while True:
            info = next(learning_it)
            if info.is_discriminator:
                driver.post_disc_update(
                    info.gen_step,
                    info.disc_step,
                    info.disc_loss,
                    info.accuracy,
                    info.gen_time,
                    info.disc_time,
                    ssnode.null_FixedPointsInfo,
                )
                last_info = info
            else:
                assert info.gen_step == k
                # Save fake and tuning curves averaged over Zs:
                data_mean = last_info.xd.mean(axis=0).tolist()
                gen_mean = last_info.xg.mean(axis=0).tolist()
                driver.datastore.tables.saverow('TC_mean.csv',
                                                gen_mean + data_mean)

                # Then the generator step was just taken.  Let's
                # return a result that GANDriver understands.
                return UpdateResult(
                    Gloss=info.gen_loss,
                    Dloss=last_info.disc_loss,
                    Daccuracy=last_info.accuracy,
                    SSsolve_time=info.gen_time,
                    gradient_time=info.disc_time,
                    model_info=ssnode.null_FixedPointsInfo,
                    rate_penalty=last_info.dynamics_penalty,
                )
                # See: [[../recorders.py::def record.*update_result]]


def make_parser():
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)

    # Dataset
    parser.add_argument(
        '--truth_size', default=1000, type=int,
        help='''Number of SSNs to be used to generate ground truth
        data (default: %(default)s)''')
    parser.add_argument(
        '--truth_seed', default=42, type=int,
        help='Seed for generating ground truth data (default: %(default)s)')

    # Generator & Discriminator trainers
    for prefix in ['gen', 'disc']:
        parser.add_argument(
            '--{}-learning-rate'.format(prefix),
            '--{}-learn-rate'.format(prefix),
            default=0.01, type=float,
            help='{} learning rate (default: %(default)s)'.format(prefix))
        parser.add_argument(
            '--{}-update-name'.format(prefix),
            default='adam-wgan',
            help='{} update method (default: %(default)s)'.format(prefix))

    # Generator
    parser.add_argument(
        '--batchsize', '--n_samples', default=15, type=eval,
        help='''Number of samples to draw from G each step
        (aka NZ, minibatch size). (default: %(default)s)''')
    parser.add_argument(
        '--sample-sites', default=[0], type=utils.csv_line(float),
        help='''Locations (offsets) of neurons to be sampled from SSN in the
        "bandwidth" space [-1, 1].  0 means the center of the
        network. (default: %(default)s)''')
    parser.add_argument(
        '--contrasts', '--contrast',
        default=[20],
        type=utils.csv_line(float),
        help='Comma separated value of floats')
    parser.add_argument(
        '--include-inhibitory-neurons', action='store_true',
        help='Sample TCs from inhibitory neurons if given.')

    for name in 'JDS':
        parser.add_argument(
            '--gen-{}-min'.format(name),
            default=1e-3, type=eval,
            help='''Lower limit of the parameter {}.
            Used only if --gen-param-type=clip.
            '''.format(name))
        parser.add_argument(
            '--gen-{}-max'.format(name),
            default=10, type=eval,
            help='''Upper limit of the parameter {}.
            Used only if --gen-param-type=clip.
            '''.format(name))

    # Generator trainer
    parser.add_argument(
        '--gen-dynamics-cost', type=float, default=1,
        help='''Cost for non-fixed point behavior of the SSN during
        the period in which the tuning curves are measured.
        (default: %(default)s)''')

    # Discriminator
    parser.add_argument(
        '--disc-layers', '--layers', default=[], type=eval,
        help='''List of numbers of units in hidden layers
        (default: %(default)s)''')
    parser.add_argument(
        '--disc-normalization', default='none', choices=('none', 'layer'),
        help='Normalization used for discriminator.')
    parser.add_argument(
        '--disc-nonlinearity', default='rectify',
        help='''Nonlinearity to be used for hidden layers.
        (default: %(default)s)''')

    # Discriminator trainer
    parser.add_argument(
        '--lipschitz-cost',
        '--WGAN_lambda',  # to be compatible with gan.py
        default=10.0, type=float,
        help='The complexity penalty for the D (default: %(default)s)')
    parser.add_argument(
        '--critic-iters-init',
        '--WGAN_n_critic0',  # to be compatible with gan.py
        default=50, type=int,
        help='First critic iterations (default: %(default)s)')
    parser.add_argument(
        '--critic-iters',
        '--WGAN_n_critic',  # to be compatible with gan.py
        default=5, type=int,
        help='Critic iterations (default: %(default)s)')

    plain_gan.add_learning_options(parser)
    parser.set_defaults(
        datastore_template='logfiles/BPTT_GAN_{layers_str}',
    )
    return parser


def init_driver(
        datastore,
        iterations, quit_JDS_threshold, quiet,
        disc_param_save_interval, disc_param_template,
        disc_param_save_on_error,
        layers,
        sample_sites, include_inhibitory_neurons,
        **run_config):
    del layers  # see [[ns\.layers]] below in main() for why

    run_config = utils.subdict_by_prefix(run_config, 'disc_')
    run_config = utils.subdict_by_prefix(run_config, 'gen_')

    run_config.update(
        sample_sites=sample_sites,
        include_inhibitory_neurons=include_inhibitory_neurons,
    )
    gan, rest = make_gan(run_config)
    driver = GANDriver(
        gan, datastore,
        iterations=iterations, quiet=quiet,
        disc_param_save_interval=disc_param_save_interval,
        disc_param_template=disc_param_template,
        disc_param_save_on_error=disc_param_save_on_error,
        quit_JDS_threshold=quit_JDS_threshold,
    )

    return dict(driver=driver,
                sample_sites=sample_sites,
                include_inhibitory_neurons=include_inhibitory_neurons,
                **rest)


def main(args=None):
    parser = make_parser()
    ns = parser.parse_args(args)

    # To make "layers_str" work, set layers and "remove" it in
    # init_driver.  See also: [[../execution.py::layers_str]]
    ns.layers = ns.disc_layers

    plain_gan.do_learning(learn, vars(ns), init_driver=init_driver,
                          script_file=__file__)


if __name__ == '__main__':
    main()