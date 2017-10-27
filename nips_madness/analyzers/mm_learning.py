from types import SimpleNamespace

from matplotlib import pyplot

from ..utils import csv_line
from .mm_loader import MomentMatchingData
from .learning import plot_gen_params


def plot_loss(data, ax=None):
    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot(data.epochs,
            data.log.learning.data[:, 1],
            label='loss')

    ax.legend(loc='best')


def plot_mm_learning(data, title_params=None):
    fig, axes = pyplot.subplots(nrows=3, ncols=3,
                                sharex=True,
                                squeeze=False, figsize=(9, 6))

    plot_loss(data, axes[0, 0])

    plot_gen_params(data, axes=axes[1, :])
    plot_gen_params(data, axes=axes[2, :],
                    yscale='log', legend=False, ylim=False)

    fig.suptitle(data.pretty_spec(title_params))
    return SimpleNamespace(
        fig=fig,
        axes=axes,
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
