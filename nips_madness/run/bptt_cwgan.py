"""
Run SSN-BPTT conditional Wasserstein GAN learning.
"""

from logging import getLogger

from . import bptt_wgan
from . import gan as plain_gan
from .. import utils
from ..drivers import BPTTcWGANDriver
from ..networks.cwgan import make_gan
from .bptt_wgan import learn

logger = getLogger(__name__)


def make_parser():
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)

    parser.add_argument(
        '--num-models', default=15, type=int,
        help='Number of SSN to be instantiated (aka NZ).')
    parser.add_argument(
        '--probes-per-model', default=1, type=int,
        help='''Number of probes to be included in the generator output
        from a single SSN.''')
    parser.add_argument(
        '--norm-probes', '--sample-sites',
        default=[0], type=utils.csv_line(float),
        help='''"Normalized" probes, i.e., locations (offsets) of
        neurons to be sampled from SSN in [-1, 1] space ("bandwidth
        coordinate").  0 means the center of the network.
        (default:%(default)s)''')

    bptt_wgan.add_bptt_common_options(parser)
    plain_gan.add_learning_options(parser)
    parser.set_defaults(
        datastore_template='logfiles/BPTT_CWGAN_{layers_str}',
    )
    return parser


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
    driver = BPTTcWGANDriver(
        gan, datastore,
        iterations=iterations, quiet=quiet,
        disc_param_save_interval=disc_param_save_interval,
        disc_param_template=disc_param_template,
        disc_param_save_on_error=disc_param_save_on_error,
        quit_JDS_threshold=quit_JDS_threshold,
    )

    return dict(driver=driver,
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
