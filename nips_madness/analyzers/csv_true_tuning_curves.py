from __future__ import print_function, division

import os

import numpy as np

from ..ssnode import DEFAULT_PARAMS, sample_tuning_curves
from .loader import load_gandata


def csv_true_tuning_curves(logpath, output, quiet, bandwidths, **kwargs):
    if not os.path.exists(output):
        os.makedirs(output)

    data = load_gandata(logpath)

    paramnames = list(DEFAULT_PARAMS) + ['dt']
    gan_params = data.params()
    ssn_params = {key: gan_params[key] for key in paramnames
                  if key in gan_params}
    for ssn_key, true_key in [('seed', 'truth_seed'),
                              ('io_type', 'true_IO_type')]:
        if true_key in gan_params:
            ssn_params[ssn_key] = gan_params[true_key]
    if not quiet:
        print("Recorded SSN parameters:")
        print(ssn_params)
    ssn_params.update(bandwidths=bandwidths, **kwargs)
    data, _ = sample_tuning_curves(**ssn_params)

    np.savetxt(os.path.join(output, 'truth.csv'), data.T,
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
        '--bandwidths',
        default=[0, 0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1],
        type=lambda x: list(map(float, x.split(','))),
        help='Comma separated value of floats')
    parser.add_argument(
        '--sample-sites',
        default=3,
        help='Number of neurons per SSN to be sampled.')
    parser.add_argument(
        '--NZ', default=1000, type=int,
        help='Number of SSNs to be sampled.')
    parser.add_argument('logpath', help='Path to SSNGAN_log_*.log file.')
    parser.add_argument('output', help='Output directory')
    ns = parser.parse_args(args)
    csv_true_tuning_curves(**vars(ns))


if __name__ == '__main__':
    main()
