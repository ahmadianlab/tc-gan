from matplotlib import pyplot
import matplotlib
import numpy as np

from ..ssnode import DEFAULT_PARAMS
from .loader import load_gandata


def plot_learning(data):
    df = data.to_dataframe()
    fig, axes = pyplot.subplots(nrows=2, ncols=2,
                                squeeze=False, figsize=(9, 6))
    df.plot('epoch', ['Gloss', 'Dloss'], ax=axes[0, 0])
    df.plot('epoch', ['Daccuracy'], ax=axes[0, 1])
    df.plot('epoch', ['SSsolve_time', 'gradient_time'], ax=axes[1, 0])
    df.plot('epoch', ['model_convergence', 'truth_convergence'], ax=axes[1, 1])
    fig.suptitle(data.tag)
    return fig


def plot_tuning_curve_evo(data, epochs=None, ax=None, cmap='inferno_r',
                          linewidth=0.3, ylim='auto',
                          xlabel='Bandwidths',
                          ylabel='Average Firing Rate'):
    if ax is None:
        _, ax = pyplot.subplots()

    start = 0
    if epochs is None:
        stop = len(data.tuning)
    elif isinstance(epochs, int):
        stop = epochs
    else:
        start, stop = epochs

    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(start, stop)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig = ax.get_figure()
    cb = fig.colorbar(mappable, ax=ax)
    cb.set_label('epochs')

    bandwidths = np.asarray(DEFAULT_PARAMS['bandwidths'])
    for i in range(start, stop):
        ax.plot(bandwidths, data.model_tuning[i], color=cmap(norm(i)),
                linewidth=linewidth)
    if ylim == 'auto':
        y = data.model_tuning[start:stop]
        q3 = np.percentile(y, 75)
        q1 = np.percentile(y, 25)
        iqr = q3 - q1
        yamp = y[y < q3 + 1.5 * iqr].max()
        ax.set_ylim(- yamp * 0.05, yamp * 1.2)
    elif ylim:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax


def analyze_learning(logpath, show, figpath):
    fig = plot_learning(load_gandata(logpath))
    if show:
        pyplot.show()
    if figpath:
        fig.savefig(figpath)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logpath', help='path to SSNGAN_log_*.log')
    parser.add_argument('--figpath')
    parser.add_argument('--show', action='store_true')
    ns = parser.parse_args(args)
    analyze_learning(**vars(ns))


if __name__ == '__main__':
    main()