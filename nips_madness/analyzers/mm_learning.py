from types import SimpleNamespace
import itertools

from matplotlib import pyplot
import numpy as np

from .learning import plot_gen_params, gen_param_smape


def plot_loss(rec, ax=None):
    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot('epoch', 'loss', label='loss', data=rec.learning)
    ax.set_yscale('log')

    ax.legend(loc='best')


def plot_moments(rec, moment, ax=None):
    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot('epoch', moment, data=rec.gen_moments, linewidth=0.1)
    ax.set_yscale('log')

    ax.text(0.05, 0.95,
            moment,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize='small',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
            transform=ax.transAxes)


def moments_smape(rec):
    gen_moments = rec.gen_moments[rec.moment_names]  # exclude 'epoch' etc.
    num = gen_moments.sub(rec.data_moments)
    den = gen_moments.add(rec.data_moments)
    return 200 * np.nanmean(np.abs(num / den), axis=1)


def plot_mm_smape(rec, ax=None, colors=0,
                  ylim=(0, 200)):
    """
    Plot sMPAE of mean TC and generator parameter.

    Parameters
    ----------
    rec : `.MomentMatchingRecords`

    """
    if ax is None:
        _, ax = pyplot.subplots()

    if isinstance(colors, int):
        colors = map('C{}'.format, itertools.count(colors))
    else:
        colors = iter(colors)

    ax.plot(rec.gen_moments['epoch'], moments_smape(rec),
            color=next(colors),
            label='Mom. sMAPE')
    ax.plot(rec.generator['epoch'], gen_param_smape(rec),
            color=next(colors),
            label='G param. sMAPE')

    ax.legend(loc='best')

    if ylim:
        ax.set_ylim(ylim)


def plot_mm_learning(rec, title_params=None):
    is_heteroin = rec.rc.ssn_type in ('heteroin', 'deg-heteroin')
    fig, axes = pyplot.subplots(nrows=3, ncols=3,
                                sharex=True,
                                squeeze=False, figsize=(9, 6))

    plot_loss(rec, axes[0, 0])
    plot_mm_smape(rec, ax=axes[0, 2])

    if is_heteroin:
        plot_gen_params(rec, axes=[axes[0, 1]] + list(axes[1, :]))
    else:
        plot_gen_params(rec, axes=axes[1, :])
    plot_gen_params(rec, axes=axes[2, :], param_array_names=['J', 'D', 'S'],
                    yscale='log', legend=False, ylim=False)

    for ax in axes[-1]:
        ax.set_xlabel('epoch')

    def add_upper_ax(ax):
        def sync_xlim(ax):
            ax_up.set_xlim(*map(epoch_to_step, ax.get_xlim()))

        epoch_to_step = rec.rc.epoch_to_step

        ax_up = ax.twiny()
        ax_up.set_xlabel('step')

        sync_xlim(ax)
        ax.callbacks.connect('xlim_changed', sync_xlim)
        return ax_up

    axes_upper = list(map(add_upper_ax, axes[0]))
    # See:
    # http://matplotlib.org/gallery/subplots_axes_and_figures/fahrenheit_celsius_scales.html
    # https://github.com/matplotlib/matplotlib/issues/7161#issuecomment-249620393

    fig.suptitle(rec.pretty_spec(title_params))
    return SimpleNamespace(
        fig=fig,
        axes=axes,
        axes_upper=axes_upper,
    )
