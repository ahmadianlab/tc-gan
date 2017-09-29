import itertools

from matplotlib import pyplot
import matplotlib
import numpy as np

from ..ssnode import DEFAULT_PARAMS
from .loader import load_gandata


def plot_errors(data, legend=True, ax=None):
    if ax is None:
        _, ax = pyplot.subplots()
    from numpy.linalg import norm
    import matplotlib.patheffects as pe

    model = data.model_tuning
    true = data.true_tuning
    per_stim_error = abs(model - true) / abs(true)
    total_error = norm(model - true, axis=-1) / norm(true, axis=-1)

    per_stim_lines = ax.plot(per_stim_error, alpha=0.4)
    total_error_lines = ax.plot(
        total_error,
        path_effects=[pe.Stroke(linewidth=5, foreground='white'),
                      pe.Normal()])

    if legend:
        leg = ax.legend(
            total_error_lines + per_stim_lines,
            ['rel. l2 error'] + list(range(len(per_stim_lines))),
            loc='center left')
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor('white')

    return dict(
        ax=ax,
        per_stim_error=per_stim_error,
        per_stim_lines=per_stim_lines,
        total_error=total_error,
        total_error_lines=total_error_lines,
    )


def plot_gen_params(data, axes=None, yscale=None, legend=True, ylim=True):
    if axes is None:
        _, axes = pyplot.subplots(ncols=3, sharex=True, figsize=(9, 3))
    for column, name in enumerate('JDS'):
        true_param = DEFAULT_PARAMS[name]
        fake_param = data.gen_param(name)
        for c, ((i, p), (j, q)) in enumerate(itertools.product(
                enumerate('EI'), enumerate('EI'))):
            color = 'C{}'.format(c)
            axes[column].axhline(
                true_param[i, j],
                linestyle='--',
                color=color)
            axes[column].plot(
                fake_param[:, i, j],
                label='${}_{{{}{}}}$'.format(name, p, q),
                color=color)
        if ylim:
            _, ymax0 = axes[column].get_ylim()
            ymax1 = true_param.max() * 2.0
            if ymax0 > ymax1:
                axes[column].set_ylim(-0.05 * ymax1, ymax1)
        if yscale:
            axes[column].set_yscale(yscale)
        if legend:
            leg = axes[column].legend(loc='best')
            leg.set_frame_on(True)
            leg.get_frame().set_facecolor('white')
    return axes


def plot_learning(data):
    df = data.to_dataframe()
    fig, axes = pyplot.subplots(nrows=4, ncols=3,
                                sharex=True,
                                squeeze=False, figsize=(9, 8))
    df.plot('epoch', ['Gloss', 'Dloss'], ax=axes[0, 0], alpha=0.8, logy=True)
    df.plot('epoch', ['Daccuracy'], ax=axes[0, 1])
    df.plot('epoch', ['SSsolve_time', 'gradient_time'], ax=axes[1, 0],
            logy=True)
    df.plot('epoch', ['model_convergence'], ax=axes[1, 1], logy=True)

    ax_loss = axes[0, 0]
    ax_loss.set_yscale('symlog')

    ax_dacc = axes[0, 1]
    ax_dacc.set_yscale('symlog')
    # q1, q3 = np.percentile(df.loc[:, 'Daccuracy'], [25, 75])
    # iqr = q3 - q1
    # linthreshy = max((df.loc[:, 'Daccuracy'] < q3 + 1.5 * iqr).max(),
    #                  -(df.loc[:, 'Daccuracy'] < q1 - 1.5 * iqr).min())
    # ax_dacc.set_yscale('symlog', linthreshy=linthreshy)

    err1 = plot_errors(data, ax=axes[0, 2])
    err2 = plot_errors(data, ax=axes[1, 2], legend=False)
    err1['ax'].set_ylim(0, 1)
    err2['ax'].set_yscale('log')

    plot_gen_params(data, axes=axes[2, :])
    plot_gen_params(data, axes=axes[3, :],
                    yscale='log', legend=False, ylim=False)

    fig.suptitle(data.pretty_spec())
    return fig


def plot_tuning_curve_evo(data, epochs=None, ax=None, cmap='inferno_r',
                          linewidth=0.3, ylim='auto',
                          include_true=True,
                          xlabel='Bandwidths',
                          ylabel='Average Firing Rate'):
    if ax is None:
        _, ax = pyplot.subplots()

    if epochs is None:
        epochs = len(data.tuning)
    elif isinstance(epochs, int):
        epochs = range(10)

    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(min(epochs), max(epochs))
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig = ax.get_figure()
    cb = fig.colorbar(mappable, ax=ax)
    cb.set_label('epochs')

    bandwidths = data.bandwidths
    for i in epochs:
        ax.plot(bandwidths, data.model_tuning[i], color=cmap(norm(i)),
                linewidth=linewidth)
    if include_true:
        ax.plot(bandwidths, data.true_tuning[0],
                linewidth=3, linestyle='--')

    if ylim == 'auto':
        y = data.model_tuning[epochs]
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
    parser.add_argument(
        'logpath',
        help='''Path to GAN output directory. It can also be any file
        in such directory; filename part is ignored.''')
    parser.add_argument('--figpath')
    parser.add_argument('--show', action='store_true')
    ns = parser.parse_args(args)
    analyze_learning(**vars(ns))


if __name__ == '__main__':
    main()
