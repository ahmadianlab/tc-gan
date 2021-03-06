"""
Generate tuning curves from generator parameters.

.. WARNING:: epochs here means (generator) update steps.

.. TODO:: consistently rename epochs to steps.

"""

from __future__ import print_function, division

import os

import numpy as np

from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..utils import make_progressbar, csv_line
from ..ssnode import DEFAULT_PARAMS
from .loader import load_learning
from .distdiff import generated_tuning_curves


def csv_tuning_curves(logpath, output, sample_epochs, quiet,
                      bandwidths, sample_sites, **kwargs):
    if not os.path.exists(output):
        os.makedirs(output)
    if isinstance(sample_epochs, str):
        sample_epochs = eval(sample_epochs, {}, vars(np))
    sample_epochs = np.asarray(sample_epochs)

    data = load_learning(logpath)

    gan_params = data.params()
    ssn_params = {key: gan_params[key] for key in DEFAULT_PARAMS
                  if key in gan_params}
    if not quiet:
        print("Recorded SSN parameters:")
        print(ssn_params)

    num_sites = ssn_params.get('N', DEFAULT_PARAMS['N'])
    ssn_params.update(
        bandwidths=bandwidths,
        sample_sites=sample_sites_from_stim_space(sample_sites, num_sites),
        **kwargs)

    sample_epochs = sample_epochs[sample_epochs < data.total_steps]
    tuning_curves = generated_tuning_curves(data, indices=sample_epochs,
                                            **ssn_params)
    bar = make_progressbar(quiet=quiet, max_value=len(sample_epochs))
    for i, (fake, _) in bar(enumerate(tuning_curves)):
        path = os.path.join(output, '{:010d}.csv'.format(i))
        np.savetxt(path, fake.T, delimiter=',')

    np.savetxt(os.path.join(output, 'bandwidths.csv'), bandwidths,
               delimiter=',')
    np.savetxt(os.path.join(output, 'sample_epochs.csv'), sample_epochs,
               delimiter=',')


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=type('FormatterClass',
                             (argparse.RawDescriptionHelpFormatter,
                              argparse.ArgumentDefaultsHelpFormatter),
                             {}),
        description=__doc__)
    parser.add_argument('--quiet', action='store_true',)
    parser.add_argument(
        '--sample-epochs',
        default='logspace(1, 14, 14, base=2, dtype=int)',
        help='Epochs at which tuning curves are sampled.')
    parser.add_argument(
        '--bandwidths',
        default=[0, 0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1],
        type=csv_line(float),
        help='Comma separated value of floats')
    parser.add_argument(
        '--sample-sites',
        default=[0],
        type=csv_line(float),
        help='''Probe offsets in [0, 1]-coordinate (bandwidth unit).
        Comma separated value of floats.''')
    parser.add_argument(
        '--contrast', '--contrasts',
        default=[20.],
        type=csv_line(float),
        help='Contrasts of stimulus.')
    parser.add_argument(
        '--NZ', default=30, type=int,
        help='Number of SSNs to be sampled.')
    parser.add_argument('logpath', help='Path to SSNGAN_log_*.log file.')
    parser.add_argument('output', help='Output directory')
    ns = parser.parse_args(args)
    csv_tuning_curves(**vars(ns))


if __name__ == '__main__':
    main()
