import itertools

from matplotlib import pyplot
import matplotlib
import numpy as np

from ..utils import Namespace, log_timing


def maybe_downsample_to(downsample_to, x):
    if downsample_to:
        stride = len(x) // downsample_to
        if stride > 1:
            return x[::stride]
    return x


def clip_ymax(ax, ymax, ymin=0):
    if ax.get_ylim()[1] > ymax:
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(ymin, None)


def smape(x, y, axis=-1):
    """
    Symmetric mean absolute percentage error (sMAPE).

    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return 200 * np.nanmean(np.abs((x - y) / (x + y)), axis=axis)
# Use nanmean to make it work for the case both x and y are zero at
# some indices.


def gen_param_smape(rec):
    return smape(rec.flatten_true_params(ndim=2),
                 rec.flatten_gen_params())


def plot_data_smape(rec, ax=None, downsample_to=None, colors=0,
                    ylim=(0, 200)):
    """
    Plot sMPAE of mean TC and generator parameter.

    Parameters
    ----------
    rec : `.GANRecords`

    """
    if ax is None:
        _, ax = pyplot.subplots()

    if isinstance(colors, int):
        colors = map('C{}'.format, itertools.count(colors))
    else:
        colors = iter(colors)

    TC_mean = maybe_downsample_to(downsample_to, rec.TC_mean)
    ax.plot(TC_mean['epoch'], smape(TC_mean['gen'], TC_mean['data']),
            color=next(colors),
            label='TC sMAPE')
    ax.plot(maybe_downsample_to(downsample_to, rec.generator['epoch']),
            maybe_downsample_to(downsample_to, gen_param_smape(rec)),
            color=next(colors),
            label='G param. sMAPE')

    ax.legend(loc='best')

    if ylim:
        ax.set_ylim(ylim)


def plot_tc_errors(rec, legend=True, ax=None, per_stim=False,
                   ylim=(0, 200)):
    """
    Plot tuning curve (TC) sMAPE.

    .. WARNING:: Untested!
    .. TODO:: Test or remove `plot_tc_errors`.

    Parameters
    ----------
    rec : `.GANRecords`

    """
    if ax is None:
        _, ax = pyplot.subplots()
    import matplotlib.patheffects as pe

    epoch = rec.TC_mean['epoch']
    model = rec.TC_mean['gen'].as_matrix()
    true = rec.TC_mean['data'].as_matrix()
    total_error = smape(model, true)

    total_error_lines = ax.plot(
        epoch,
        total_error,
        path_effects=[pe.Stroke(linewidth=5, foreground='white'),
                      pe.Normal()])
    if per_stim:
        per_stim_error = 200 * abs((model - true) / (model + true))
        per_stim_lines = ax.plot(epoch, per_stim_error, alpha=0.4)
    else:
        per_stim_error = per_stim_lines = None

    if legend:
        if per_stim:
            leg = ax.legend(
                total_error_lines + per_stim_lines,
                ['TC sMAPE'] + list(range(len(per_stim_lines))),
                loc='center left')
        else:
            leg = ax.legend(
                total_error_lines,
                ['TC sMAPE'],
                loc='upper left')
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor('white')

    if ylim:
        ax.set_ylim(ylim)

    return Namespace(
        ax=ax,
        per_stim_error=per_stim_error,
        per_stim_lines=per_stim_lines,
        total_error=total_error,
        total_error_lines=total_error_lines,
    )


def name_to_tex(name):
    if '_' in name:
        return '${}_{{{}}}$'.format(*name.split('_', 1))
    else:
        return name


def plot_gen_params(rec, axes=None, yscale=None, legend=True, ylim=True,
                    downsample_to=None,
                    param_array_names=None):
    """
    Plot evolution of generator parameters.

    Generator parameters in `rec.generator <.BaseRecords.generator>`
    and corresponding "true" parameters are plotted.

    Parameters
    ----------
    rec : `.BaseRecords`
        Works for both GAN and moment-matching data records.

    """
    if param_array_names is None:
        param_array_names = rec.param_array_names
    if axes is None:
        ncols = len(param_array_names)
        _, (axes,) = pyplot.subplots(ncols=ncols, sharex=True, squeeze=False,
                                     figsize=(3 * ncols, 3))
    else:
        if len(axes) != len(param_array_names):
            raise ValueError('Needs {} axes; {} given.'
                             .format(len(param_array_names), len(axes)))

    generator = maybe_downsample_to(downsample_to, rec.generator)
    epoch = generator['epoch']

    arts = {}
    arts['gen_lines'] = gen_lines = {}
    arts['true_lines'] = true_lines = {}
    for ax, array_name in zip(axes, param_array_names):
        element_names = [name for name in rec.param_element_names
                         if name.startswith(array_name)]
        for c, name in enumerate(element_names):
            color = 'C{}'.format(c)
            true_lines[name] = ax.axhline(
                rec.rc.get_true_param(name),
                linestyle='--',
                color=color)
            gen_lines[name] = ax.plot(
                epoch,
                generator[name],
                label=name_to_tex(name),
                color=color)

        if ylim:
            _, ymax0 = ax.get_ylim()
            ymax1 = max(map(rec.rc.get_true_param, element_names)) * 2.0
            if ymax0 > ymax1:
                ax.set_ylim(-0.05 * ymax1, ymax1)
        if yscale:
            ax.set_yscale(yscale)
        if legend:
            leg = ax.legend(loc='best')
            leg.set_frame_on(True)
            leg.get_frame().set_facecolor('white')

    return arts


def plot_gan_cost_and_rate_penalty(rec, ax=None, downsample_to=None,
                                   ymax_dacc=1, ymin_dacc=-0.05,
                                   yscale_dacc='symlog',
                                   yscale_rate_penalty='log'):
    """
    Plot GAN loss and rate penalty (if recorded).

    Parameters
    ----------
    rec : `.GANRecords`
        `rec.learning <.GANRecords.learning>` is plotted.
        If it is a WGAN, Wasserstein distance is plotted.

    """
    if ax is None:
        _, ax = pyplot.subplots()
    df = rec.learning
    df = maybe_downsample_to(downsample_to, df)

    color = 'C0'
    if rec.rc.is_WGAN:
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

    ax.set_yscale(yscale_dacc)


def disc_param_stats_to_pretty_label(name):
    """
    Convert param_stats names to latex label for plotting.

    >>> disc_param_stats_to_pretty_label('W.nnorm')
    '$|W_0|$'
    >>> disc_param_stats_to_pretty_label('W.nnorm.1')
    '$|W_1|$'
    >>> disc_param_stats_to_pretty_label('b.nnorm.2')
    '$|b_2|$'
    >>> disc_param_stats_to_pretty_label('scales.nnorm.3')
    '$|g_3|$'
    >>> disc_param_stats_to_pretty_label('spam')

    """
    parts = name.split('.')
    if 2 <= len(parts) <= 3 and parts[1] == 'nnorm':
        var = parts[0]
        suffix = parts[2] if len(parts) == 3 else '0'
        if var == 'scales':
            var = 'g'
            # "g" for gain; see: Ba et al (2016) Layer Normalization
        return '$|{}_{}|$'.format(var, suffix)


def plot_disc_param_stats(
        rec, ax=None, logy=True, downsample_to=None,
        legend=dict(loc='center left', ncol='auto', fontsize='small',
                    handlelength=0.5, columnspacing=0.4),
        legend_max_rows=7,
        **kwargs):
    """
    Plot recorded statistics of discriminator network parameters.

    Parameters
    ----------
    rec : `.GANRecords`
        `rec.disc_param_stats <.GANRecords.disc_param_stats>` is plotted.

    """
    if ax is None:
        _, ax = pyplot.subplots()
    param_names = rec.disc_param_stats_names
    maybe_downsample_to(downsample_to, rec.disc_param_stats).plot(
        'epoch', param_names, logy=logy, legend=False, ax=ax, **kwargs)

    for line in ax.get_lines():
        label = disc_param_stats_to_pretty_label(line.get_label())
        if label:
            line.set_label(label)

    if legend:
        if not isinstance(legend, dict):
            legend = {}
        legend = dict(legend)
        if legend.get('ncol') == 'auto':
            n_stats = len(param_names)
            legend['ncol'] = int(np.ceil(n_stats / legend_max_rows))
        leg = ax.legend(**legend)
        leg.set_frame_on(True)


def plot_learning(rec, title_params=None, downsample_to=None):
    """
    Plot various GAN learning records are plotted in 4x3 axes.

    Parameters
    ----------
    rec : `.GANRecords`

    """
    common = dict(downsample_to=downsample_to)
    df = maybe_downsample_to(downsample_to, rec.learning).copy()
    fig, axes = pyplot.subplots(nrows=4, ncols=3,
                                sharex=True,
                                squeeze=False, figsize=(9, 8))
    is_heteroin = rec.rc.ssn_type in ('heteroin', 'deg-heteroin')

    plot_kwargs = dict(ax=axes[0, 0], alpha=0.8)
    if rec.rc.is_WGAN:
        df['Lip. penalty'] = df['Dloss'] - df['Daccuracy']
        df.plot('epoch', ['Gloss', 'Dloss', 'Lip. penalty'], **plot_kwargs)
    else:
        df.plot('epoch', ['Gloss', 'Dloss'], **plot_kwargs)

    plot_gan_cost_and_rate_penalty(rec, ax=axes[0, 1], **common)

    df.plot('epoch', ['SSsolve_time', 'gradient_time'], ax=axes[1, 0],
            logy=True)
    if not is_heteroin:
        df.plot('epoch', ['model_convergence'], ax=axes[1, 1], logy=True)

    ax_loss = axes[0, 0]
    ax_loss.set_yscale('symlog')

    plot_data_smape(rec, ax=axes[0, 2], colors=2, **common)

    plot_disc_param_stats(rec, ax=axes[1, 2], **common)

    if is_heteroin:
        plot_gen_params(rec, axes=[axes[1, 1]] + list(axes[2, :]), **common)
    else:
        plot_gen_params(rec, axes=axes[2, :], **common)
    plot_gen_params(rec, axes=axes[3, :], param_array_names=['J', 'D', 'S'],
                    yscale='log', legend=False, ylim=False, **common)

    for ax in axes[-1]:
        ax.set_xlabel('epoch')

    def add_upper_ax(ax):
        def sync_xlim(ax):
            ax_up.set_xlim(*map(epoch_to_gen_step, ax.get_xlim()))

        epoch_to_gen_step = rec.rc.epoch_to_gen_step

        ax_up = ax.twiny()
        ax_up.set_xlabel('gen_step')

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


def plot_tuning_curve_evo(data, epochs=None, ax=None, cmap='inferno_r',
                          linewidth=0.3, ylim='auto',
                          include_true=True,
                          xlabel='Bandwidths',
                          ylabel='Average Firing Rate'):
    """
    Plot evolution of TC averaged over noise (zs).

    .. WARNING:: It is not used for a long time.
    .. TODO:: Make `plot_tuning_curve_evo` accept `.GANRecords`.

    Parameters
    ----------
    data : `.GANData`

    """
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


def analyze_learning(logpath, title_params, downsample_to, force_load):
    """
    Load learning records from `logpath` and plot them in a figure.
    """
    from ..loaders import load_records
    rec = load_records(logpath)
    if force_load:
        with log_timing('{}.load_all()'.format(rec)):
            rec.load_all()
    if rec.run_module == 'bptt_moments':
        from .mm_learning import plot_mm_learning
        with log_timing('plot_mm_learning()'):
            return plot_mm_learning(rec, title_params, downsample_to)
    else:
        with log_timing('plot_learning()'):
            return plot_learning(rec, title_params, downsample_to)


def cli_analyze_learning(*args, **kwargs):
    arts = analyze_learning(*args, **kwargs)
    return arts.fig


def main(args=None):
    from .basecli import make_base_parser, call_cli
    parser = make_base_parser(description=__doc__)
    parser.add_argument(
        '--force-load', action='store_true',
        help='Run rec.load_all() before plot.  Useful for benchmarking'
        ' (with --log-level=DEBUG).')
    parser.add_argument(
        '--downsample-to', type=int, default=None, metavar='<num>',
        help='Down-sample records to have at least <num> data points.')
    call_cli(cli_analyze_learning, parser.parse_args(args))


if __name__ == '__main__':
    main()
