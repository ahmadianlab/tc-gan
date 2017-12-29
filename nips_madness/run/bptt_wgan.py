"""
Run SSN-BPTT Wasserstein GAN learning.
"""

from logging import getLogger

import numpy as np

from . import gan as plain_gan
from .. import utils
from ..drivers import BPTTWGANDriver
from ..networks.wgan import make_gan, DEFAULT_PARAMS
from ..networks.dataset import generate_dataset

logger = getLogger(__name__)


def generate_dataset_and_save(datastore, learner, **kwargs):
    data = generate_dataset(learner, **kwargs)

    with utils.log_timing('numpy.save("truth.npy", data)'):
        np.save(datastore.path('truth.npy'), data)
    # Note: saving in .npy rather than .npz so that it can be
    # deduplicated (by, e.g., git-annex).

    return data


def learn(driver, **generate_dataset_kwargs):

    np.random.seed(0)
    gan = driver.gan
    logger.info('Compiling Theano functions...')
    with utils.log_timing('gan.prepare()'):
        gan.prepare()

    data = generate_dataset_and_save(
        driver.datastore, gan,
        **generate_dataset_kwargs)

    gan.set_dataset(data)
    driver.run(gan)


def make_parser():
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)

    parser.add_argument(
        '--batchsize', '--n_samples', default=15, type=eval,
        help='''Number of samples to draw from G each step
        (aka NZ, minibatch size). (default: %(default)s)''')
    parser.add_argument(
        '--sample-sites', default=[0], type=utils.csv_line(float),
        help='''Locations (offsets) of neurons to be sampled from SSN in the
        "bandwidth" space [-1, 1].  0 means the center of the
        network. (default: %(default)s)''')

    add_bptt_common_options(parser)
    plain_gan.add_learning_options(parser)
    parser.set_defaults(
        datastore_template='logfiles/BPTT_WGAN_{layers_str}',
    )
    return parser


def add_bptt_common_options(parser):
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
        '--seqlen', default=DEFAULT_PARAMS['seqlen'], type=int,
        help='Total time steps for SSN.')
    parser.add_argument(
        '--skip-steps', default=DEFAULT_PARAMS['skip_steps'], type=int,
        help='''First time steps to be excluded from tuning curve and
        dynamics penalty calculations.''')
    parser.add_argument(
        '--contrasts', '--contrast',
        default=[20],
        type=utils.csv_line(float),
        help='Comma separated value of floats')
    parser.add_argument(
        '--include-inhibitory-neurons', action='store_true',
        help='Sample TCs from inhibitory neurons if given.')
    parser.add_argument(
        '--unroll-scan', action='store_true',
        help='''Unroll recurrent steps for SSN.  It may make SSN
        forward/backward computations time-efficient at the cost of
        increased memory consumption.
        See: lasagne.layers.CustomRecurrentLayer''')

    for name in 'JDS':
        parser.add_argument(
            '--gen-{}-min'.format(name),
            default=1e-3, type=float,
            help='''Lower limit of the parameter {}.
            '''.format(name))
        parser.add_argument(
            '--gen-{}-max'.format(name),
            default=10, type=float,
            help='''Upper limit of the parameter {}.
            '''.format(name))
        parser.add_argument(
            '--{}0'.format(name),
            default=0.01, type=eval,
            help='''Initial value the parameter {} of the generator.
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


def init_driver(
        datastore,
        iterations, quit_JDS_threshold, quiet,
        disc_param_save_interval, disc_param_template,
        disc_param_save_on_error,
        layers,
        **run_config):
    del layers  # see [[ns\.layers]] below in main() for why

    run_config = utils.subdict_by_prefix(run_config, 'disc_')
    run_config = utils.subdict_by_prefix(run_config, 'gen_')

    gan, rest = make_gan(run_config)
    driver = BPTTWGANDriver(
        gan, datastore,
        iterations=iterations, quiet=quiet,
        disc_param_save_interval=disc_param_save_interval,
        disc_param_template=disc_param_template,
        disc_param_save_on_error=disc_param_save_on_error,
        quit_JDS_threshold=quit_JDS_threshold,
    )

    return dict(driver=driver,
                **rest)


def preprocess(run_config):
    """
    Pre-process `run_config` before it is dumped to ``info.json``.
    """
    plain_gan.preprocess(run_config)
    if run_config.get('ssn_type') == 'heteroin':
        run_config['true_ssn_options'].setdefault('V', [0.3, 0])
    elif run_config.get('ssn_type') == 'deg-heteroin':
        run_config['true_ssn_options'].setdefault('V', 0.5)


def do_learning(learn, run_config, script_file, init_driver,
                preprocess=preprocess, **kwargs):
    plain_gan.do_learning(
        learn, run_config, script_file, init_driver,
        preprocess=preprocess, **kwargs)


def main(args=None):
    parser = make_parser()
    ns = parser.parse_args(args)

    # To make "layers_str" work, set layers and "remove" it in
    # init_driver.  See also: [[../execution.py::layers_str]]
    ns.layers = ns.disc_layers

    do_learning(learn, vars(ns), init_driver=init_driver,
                script_file=__file__)


if __name__ == '__main__':
    main()
