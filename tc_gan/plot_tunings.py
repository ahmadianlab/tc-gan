from __future__ import print_function, division

from matplotlib import pyplot
import numpy as np

from .ssnode import sample_tuning_curves, DEFAULT_PARAMS


def plot_tunings(bandwidths=DEFAULT_PARAMS['bandwidths'],
                 sample_sites=3,
                 track_offset_identity=False,
                 linewidth=0.2, ylim=(None, None),
                 **sample_kwargs):
    tunings, sample = sample_tuning_curves(
        bandwidths=bandwidths,
        sample_sites=sample_sites,
        track_offset_identity=track_offset_identity,
        **sample_kwargs)

    xlabel = 'Bandwidths'
    if track_offset_identity:
        bandwidths = np.asarray(bandwidths)
        bandwidths = np.concatenate([
            bandwidths + i for i in range(sample_sites)
        ])
        xlabel = 'Bandwidths (shifted for different sites)'

    fig, ax = pyplot.subplots()
    ax.plot(bandwidths, tunings, color='C0', linewidth=linewidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Rates')
    ax.set_ylim(*ylim)

    return locals()


def main(args=None):
    plot_tunings()
    pyplot.show()


if __name__ == '__main__':
    main()
