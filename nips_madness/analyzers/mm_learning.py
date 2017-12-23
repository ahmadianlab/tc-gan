from types import SimpleNamespace

from matplotlib import pyplot

from ..utils import csv_line
from .mm_loader import MomentMatchingData
from .learning import plot_gen_params


def plot_loss(data, ax=None):
    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot(data.epochs,
            data.learning[:, 1],
            label='loss')
    ax.set_yscale('log')

    ax.legend(loc='best')


def plot_moments(data, moment, ax=None):
    if ax is None:
        _, ax = pyplot.subplots()

    imom = {
        'mean': 0,
        'var': 1,
    }[moment]

    gen_moments = data.gen_moments[:, imom]
    ax.plot(data.epochs, gen_moments, linewidth=0.1)
    ax.set_yscale('log')

    ax.text(0.05, 0.95,
            moment,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize='small',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
            transform=ax.transAxes)


def plot_mm_learning(data, title_params=None):
    raise NotImplementedError
    fig, axes = pyplot.subplots(nrows=3, ncols=3,
                                sharex=True,
                                squeeze=False, figsize=(9, 6))

    plot_loss(data, axes[0, 0])
    plot_moments(data, 'mean', axes[0, 1])
    plot_moments(data, 'var', axes[0, 2])

    plot_gen_params(data, axes=axes[1, :])
    plot_gen_params(data, axes=axes[2, :],
                    yscale='log', legend=False, ylim=False)

    for ax in axes[-1]:
        ax.set_xlabel('epoch')

    def add_upper_ax(ax):
        def sync_xlim(ax):
            ax_up.set_xlim(*map(data.epoch_to_step, ax.get_xlim()))

        ax_up = ax.twiny()
        ax_up.set_xlabel('step')

        sync_xlim(ax)
        ax.callbacks.connect('xlim_changed', sync_xlim)
        return ax_up

    axes_upper = list(map(add_upper_ax, axes[0]))
    # See:
    # http://matplotlib.org/gallery/subplots_axes_and_figures/fahrenheit_celsius_scales.html
    # https://github.com/matplotlib/matplotlib/issues/7161#issuecomment-249620393

    fig.suptitle(data.pretty_spec(title_params))
    return SimpleNamespace(
        fig=fig,
        axes=axes,
        axes_upper=axes_upper,
    )


def cli_mm_learning(datastore, show, figpath, title_params):
    data = MomentMatchingData(datastore)
    arts = plot_mm_learning(data, title_params)
    fig = arts.fig
    if show:
        pyplot.show()
    if figpath:
        fig.savefig(figpath)


def main(args=None):
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)

    parser.add_argument(
        'datastore',
        help='''Path to moment matching output directory. It can also
        be any file in such directory; filename part is ignored.''')
    parser.add_argument('--figpath')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--title-params', type=csv_line(str),
        help='Comma separated name of parameters to be used in figure title.')

    ns = parser.parse_args(args)
    cli_mm_learning(**vars(ns))


if __name__ == '__main__':
    main()
