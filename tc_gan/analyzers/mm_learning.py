import itertools

from matplotlib import pyplot
import numpy as np

from ..utils import Namespace
from .learning import plot_gen_params, gen_param_smape, maybe_downsample_to


def plot_loss(rec, ax=None, downsample_to=None, yscale_rate_penalty='log'):
    if ax is None:
        _, ax = pyplot.subplots()

    df = maybe_downsample_to(downsample_to, rec.learning)
    lines = ax.plot('epoch', 'loss', label='loss', data=df)
    ax.set_yscale('log')

    for key in ['rate_penalty', 'dynamics_penalty']:
        if key in df:
            color = 'C1'
            ax_rate_penalty = ax.twinx()
            lines += ax_rate_penalty.plot(
                'epoch', key, data=df,
                label=key, color=color, alpha=0.8)
            ax_rate_penalty.tick_params('y', colors=color)
            ax_rate_penalty.set_yscale(yscale_rate_penalty)
            break

    ax.legend(
        lines, [l.get_label() for l in lines],
        loc='best')


def plot_moments(rec, moment, ax=None, downsample_to=None):
    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot('epoch', moment, linewidth=0.1,
            data=maybe_downsample_to(downsample_to, rec.gen_moments))
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


def plot_mm_smape(rec, ax=None, downsample_to=None, colors=0,
                  ylim=(0, 200), legend=True):
    """
    Plot sMPAE of mean TC and generator parameter.

    Parameters
    ----------
    rec : `.MomentMatchingRecords`

    """
    if ax is None:
        _, ax = pyplot.subplots()
    arts = Namespace(ax=ax)

    if isinstance(colors, int):
        colors = map('C{}'.format, itertools.count(colors))
    else:
        colors = iter(colors)

    arts.lines_moments = ax.plot(
        maybe_downsample_to(downsample_to, rec.gen_moments['epoch']),
        maybe_downsample_to(downsample_to, moments_smape(rec)),
        color=next(colors),
        label='Mom. sMAPE')
    arts.lines_tc_mean = ax.plot(
        maybe_downsample_to(downsample_to, rec.generator['epoch']),
        maybe_downsample_to(downsample_to, gen_param_smape(rec)),
        color=next(colors),
        label='G param. sMAPE')
    arts.lines = arts.lines_moments + arts.lines_tc_mean

    if legend:
        ax.legend(loc='best')

    if ylim:
        ax.set_ylim(ylim)

    return arts


def plot_mm_learning(rec, title_params=None, downsample_to=None):
    common = dict(downsample_to=downsample_to)
    is_heteroin = rec.rc.ssn_type in ('heteroin', 'deg-heteroin')
    fig, axes = pyplot.subplots(nrows=3, ncols=3,
                                sharex=True,
                                squeeze=False, figsize=(9, 6))

    plot_loss(rec, axes[0, 0], **common)
    plot_mm_smape(rec, ax=axes[0, 2], **common)

    if is_heteroin:
        plot_gen_params(rec, axes=[axes[0, 1]] + list(axes[1, :]), **common)
    else:
        plot_gen_params(rec, axes=axes[1, :], **common)
    plot_gen_params(rec, axes=axes[2, :], param_array_names=['J', 'D', 'S'],
                    yscale='log', legend=False, ylim=False, **common)

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

    fig.suptitle(rec.pretty_spec(title_params, tex=True))
    return Namespace(
        fig=fig,
        axes=axes,
        axes_upper=axes_upper,
    )
