from __future__ import print_function, division

from matplotlib import pyplot

from .ssnode import sample_tuning_curves, DEFAULT_PARAMS


def plot_tunings(bandwidths=DEFAULT_PARAMS['bandwidths'],
                 linewidth=0.2, ylim=(None, None),
                 **sample_kwargs):
    sample_kwargs.update(bandwidths=bandwidths)
    tunings, sample = sample_tuning_curves(**sample_kwargs)

    fig, ax = pyplot.subplots()
    ax.plot(bandwidths, tunings, color='C0', linewidth=linewidth)
    ax.set_xlabel('Bandwidths')
    ax.set_ylabel('Rates')
    ax.set_ylim(*ylim)

    return locals()


def main(args=None):
    plot_tunings()
    pyplot.show()


if __name__ == '__main__':
    main()
