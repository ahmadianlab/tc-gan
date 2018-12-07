import itertools

from . import gaiting
from ..utils import Namespace
from .learning import maybe_downsample_to


def plot_smape_and_gaits(rec, gait_funcs=[gaiting.smr1e, gaiting.pw1e],
                         lookbacks=[1], smooths=[1, 0.25, 0],
                         rollings=['mean'],
                         yscale='log', downsample_to=1000,
                         smape_kwargs={},
                         legend_kwargs=dict(ncol=2, loc='best'),
                         ax=None):
    if isinstance(ax, (tuple, list)):
        ax_smape, ax = ax
    else:
        if ax is None:
            from matplotlib import pyplot
            _, ax = pyplot.subplots(figsize=(9.4, 4))
        ax_smape = ax
        ax = ax.twinx()

    colors = itertools.cycle(map('C{}'.format, range(10)))
    arts_smape = rec.plot_smape(ax=ax_smape, colors=colors, legend=False,
                                **smape_kwargs)
    ax_smape.set_ylabel('sMPAE')
    ax_smape.grid(False)
    for l in arts_smape.lines:
        l.set_alpha(0.5)

    gp = rec.flatten_gen_params()
    lines = []
    for gait in gait_funcs:
        for lookback_rate in lookbacks:
            lookback_step = int(lookback_rate / rec.rc.gen_learning_rate)
            for smooth_rate in smooths:
                smooth_step = int(smooth_rate / rec.rc.gen_learning_rate)
                lost_steps = lookback_step + smooth_step - 1
                xs = rec.generator['epoch'].iloc[lost_steps:]
                for stat in rollings:
                    ys = gait(gp, lookback_step, smooth_step, stat)
                    ys /= lookback_rate
                    lines += ax.plot(
                        maybe_downsample_to(downsample_to, xs),
                        maybe_downsample_to(downsample_to, ys),
                        label='{} {} bck={} sth={}'.format(
                            gait.__name__, stat, lookback_rate, smooth_rate),
                        color=next(colors))

    if yscale:
        ax.set_yscale(yscale)
    ax.set_ylabel('smoothed gait')
    ax.set_xlabel('epoch')

    all_lines = lines + arts_smape.lines
    ax.legend(all_lines, [l.get_label() for l in all_lines],
              **legend_kwargs)

    return Namespace(
        ax_gaits=ax,
        ax_smape=ax_smape,
        smape=arts_smape,
        lines=lines,
    )
