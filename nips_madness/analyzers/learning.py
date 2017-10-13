import itertools

from matplotlib import pyplot
import matplotlib
import numpy as np

from ..ssnode import DEFAULT_PARAMS
from ..utils import csv_line
from .loader import load_gandata


def clip_ymax(ax, ymax, ymin=0):
    if ax.get_ylim()[1] > ymax:
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(ymin, None)


def gen_param_mean_relative_error(data):
    true = data.true_JDS().reshape((1, -1))
    fake = data.fake_JDS()
    return np.abs((fake - true) / true).mean(axis=-1)


def plot_tc_errors(data, legend=True, ax=None, per_stim=False):
    """Plot tuning curve (TC) relative mean absolute error (MAE)."""
    if ax is None:
        _, ax = pyplot.subplots()
    import matplotlib.patheffects as pe

    model = data.model_tuning
    true = data.true_tuning
    total_error = np.abs((model - true) / true).mean(axis=-1)

    total_error_lines = ax.plot(
        total_error,
        path_effects=[pe.Stroke(linewidth=5, foreground='white'),
                      pe.Normal()])
    if per_stim:
        per_stim_error = abs(model - true) / abs(true)
        per_stim_lines = ax.plot(per_stim_error, alpha=0.4)
    else:
        per_stim_error = per_stim_lines = None

    if legend:
        if per_stim:
            leg = ax.legend(
                total_error_lines + per_stim_lines,
                ['TC rel. MAE'] + list(range(len(per_stim_lines))),
                loc='center left')
        else:
            leg = ax.legend(
                total_error_lines,
                ['TC rel. MAE'],
                loc='upper left')
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


def plot_gan_cost_and_rate_penalty(data, df=None, ax=None,
                                   ymax_dacc=1, ymin_dacc=-0.05,
                                   yscale_dacc='symlog',
                                   yscale_rate_penalty='log'):
    if ax is None:
        _, ax = pyplot.subplots()
    if df is None:
        df = data.to_dataframe()

    color = 'C0'
    if data.gan_type == 'WGAN':
        lines = ax.plot(
            df['epoch'], -df['Daccuracy'],
            label='Wasserstein distance', color=color)

        ymin0, ymax0 = ax.get_ylim()
        ymax = ymax_dacc if ymax0 < ymax_dacc else None
        ymin = ymin_dacc if ymin0 > ymin_dacc else None
        if not (ymax is None and ymin is None):
            ax.set_ylim(ymin, ymax)
    else:
        lines = ax.plot(
            'epoch', 'Daccuracy', data=df,
            label='Daccuracy', color=color)
    ax.tick_params('y', colors=color)

    if 'rate_penalty' in df:
        color = 'C1'
        ax_rate_penalty = ax.twinx()
        lines += ax_rate_penalty.plot(
            'epoch', 'rate_penalty', data=df,
            label='rate_penalty', color=color, alpha=0.8)
        ax_rate_penalty.tick_params('y', colors=color)
        ax_rate_penalty.set_yscale(yscale_rate_penalty)

    ax.legend(
        lines, [l.get_label() for l in lines],
        loc='best')

    ax.set_yscale(yscale_dacc)


def plot_learning(data, title_params=None):
    df = data.to_dataframe()
    fig, axes = pyplot.subplots(nrows=4, ncols=3,
                                sharex=True,
                                squeeze=False, figsize=(9, 8))

    plot_kwargs = dict(ax=axes[0, 0], alpha=0.8)
    if data.gan_type == 'WGAN':
        df['Lip. penalty'] = df['Dloss'] - df['Daccuracy']
        df.plot('epoch', ['Gloss', 'Dloss', 'Lip. penalty'], **plot_kwargs)
    else:
        df.plot('epoch', ['Gloss', 'Dloss'], **plot_kwargs)

    plot_gan_cost_and_rate_penalty(data, df=df, ax=axes[0, 1])

    df.plot('epoch', ['SSsolve_time', 'gradient_time'], ax=axes[1, 0],
            logy=True)
    df.plot('epoch', ['model_convergence'], ax=axes[1, 1], logy=True)

    ax_loss = axes[0, 0]
    ax_loss.set_yscale('symlog')

    err1 = plot_tc_errors(data, ax=axes[0, 2])
    clip_ymax(err1['ax'], 2)

    axes[1, 2].plot(df['epoch'], gen_param_mean_relative_error(data),
                    label='G param. rel. MAE')
    axes[1, 2].legend(loc='best')
    # clip_ymax(axes[1, 2], 1)

    plot_gen_params(data, axes=axes[2, :])
    plot_gen_params(data, axes=axes[3, :],
                    yscale='log', legend=False, ylim=False)

    for ax in axes[-1]:
        ax.set_xlabel('epoch')

    fig.suptitle(data.pretty_spec(title_params))
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


def analyze_learning(logpath, show, figpath, title_params):
    fig = plot_learning(load_gandata(logpath), title_params)
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
    parser.add_argument(
        '--title-params', type=csv_line(str),
        help='Comma separated name of parameters to be used in figure title.')
    ns = parser.parse_args(args)
    analyze_learning(**vars(ns))


if __name__ == '__main__':
    main()
