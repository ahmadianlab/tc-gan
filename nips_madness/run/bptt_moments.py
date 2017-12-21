"""
Run BTTP-based moment matching.
"""

from logging import getLogger

import numpy as np

from .bptt_wgan import generate_dataset_and_save, do_learning
from .. import execution
from .. import utils
from ..drivers import MomentMatchingDriver
from ..networks.moment_matching import make_moment_matcher, DEFAULT_PARAMS

logger = getLogger(__name__)


def learn(driver, **generate_dataset_kwargs):

    np.random.seed(0)
    mmatcher = driver.mmatcher
    logger.info('Compiling Theano functions...')
    with utils.log_timing('gan.prepare()'):
        mmatcher.prepare()

    data = generate_dataset_and_save(
        driver.datastore, mmatcher,
        **generate_dataset_kwargs)

    mmatcher.set_dataset(data)
    driver.run(mmatcher)


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

    # Driver
    parser.add_argument(
        '--iterations', default=100000, type=int,
        help='Number of iterations (default: %(default)s)')
    parser.add_argument(
        '--quiet', action='store_true',
        help='Do not print loss values per epoch etc.')

    # Generator
    parser.add_argument(
        '--batchsize', '--n_samples', default=15, type=eval,
        help='''Number of samples to draw from G each step
        (aka NZ, minibatch size). (default: %(default)s)''')
    parser.add_argument(
        '--seqlen', default=DEFAULT_PARAMS['seqlen'], type=int,
        help='Total time steps for SSN.')
    parser.add_argument(
        '--skip-steps', default=DEFAULT_PARAMS['skip_steps'], type=int,
        help='''First time steps to be excluded from tuning curve and
        dynamics penalty calculations.''')
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
    parser.add_argument(
        '--unroll-scan', action='store_true',
        help='''Unroll recurrent steps for SSN.  It may make SSN
        forward/backward computations time-efficient at the cost of
        increased memory consumption.
        See: lasagne.layers.CustomRecurrentLayer''')

    for name in 'JDS':
        parser.add_argument(
            '--{}-min'.format(name),
            default=1e-3, type=float,
            help='''Lower limit of the parameter {}.
            '''.format(name))
        parser.add_argument(
            '--{}-max'.format(name),
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
        '--lam', default=.1, type=float,
        help='Weight for the variance (default: %(default)s)')
    parser.add_argument(
        '--moment-weights-regularization', default=1e-3, type=float,
        help='Regularizer to avoid division by zero (default: %(default)s)')
    parser.add_argument(
        '--learning-rate',
        default=0.01, type=float,
        help='learning rate (default: %(default)s)')
    parser.add_argument(
        '--update-name',
        default='adam-wgan',
        help='update method (default: %(default)s)')
    parser.add_argument(
        '--dynamics-cost', type=float, default=1,
        help='''Cost for non-fixed point behavior of the SSN during
        the period in which the tuning curves are measured.
        (default: %(default)s)''')

    # Arguments handled in `.gan.preprocess`:
    parser.add_argument(
        '--n_bandwidths', default=4, type=int, choices=(1, 4, 5, 8),
        help='Number of bandwidths (default: %(default)s)')
    parser.add_argument(
        '--load-gen-param',
        help='''Path to generator.csv whose last row is loaded as the
        starting point.''')
    # TODO: Move above into `execution` module.  This file shouldn't
    #       import anything from `.run.gan`!

    execution.add_base_learning_options(parser)

    parser.set_defaults(
        datastore_template='logfiles/BPTT_MM_{lam}',
    )
    return parser


def init_driver(
        datastore,
        iterations, quiet,
        **run_config):

    mmatcher, rest = make_moment_matcher(run_config)
    driver = MomentMatchingDriver(
        mmatcher, datastore,
        iterations=iterations,
        quiet=quiet,
    )

    return dict(driver=driver,
                **rest)


def main(args=None):
    parser = make_parser()
    ns = parser.parse_args(args)

    do_learning(learn, vars(ns), init_driver=init_driver,
                script_file=__file__)


if __name__ == '__main__':
    main()
