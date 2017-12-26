from types import SimpleNamespace

from matplotlib import pyplot
import numpy as np

from ..utils import make_progressbar, add_arguments_from_function


def plot_gridified_truth(truth_df,
                         col='cell_type', row='norm_probe',
                         hue='contrast', x='bandwidth', y='rate',
                         downsample_to=None, legend_ax_idx=(-1, 0),
                         size=2, linewidth=0.1, alpha=0.5,
                         colors=None, title=None, tight_layout=False):

    icol = truth_df.columns.names.index(col)
    irow = truth_df.columns.names.index(row)
    ihue = truth_df.columns.names.index(hue)
    ixax = truth_df.columns.names.index(x)
    assert icol >= 0 and irow >= 0 and ihue >= 0 and ixax >= 0
    assert truth_df.columns.nlevels == 4

    nrows = len(truth_df.columns.levels[irow])
    ncols = len(truth_df.columns.levels[icol])
    nhues = len(truth_df.columns.levels[ihue])
    fig, axes = pyplot.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                                figsize=(size * ncols, size * nrows))

    xs = truth_df.columns.levels[ixax]
    lines = np.zeros((nrows, ncols, nhues), dtype=object)
    bar_it = iter(make_progressbar()(range(nrows * ncols * nhues)))
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if colors:
                colors_iter = iter(colors)
            else:
                colors_iter = iter(map('C{}'.format, range(nhues)))
            for k, hue_val in enumerate(truth_df.columns.levels[ihue]):

                # Constructing label-based slice (Note: iloc requires
                # flat slice even for hierarchical index.)
                idx = [slice(None)] * truth_df.columns.nlevels
                idx[irow] = truth_df.columns.levels[irow][i]
                idx[icol] = truth_df.columns.levels[icol][j]
                idx[ihue] = hue_val
                idx = tuple(idx)

                sub_df = truth_df.loc[:, idx]
                ys = sub_df.as_matrix().T
                if downsample_to:
                    ys = ys[:, :downsample_to]
                lines[i, j, k] = ax.plot(
                    xs, ys, color=next(colors_iter),
                    label='{}={}'.format(hue, hue_val),
                    linewidth=linewidth, alpha=alpha)

                next(bar_it)

            row_val = idx[irow]
            col_val = idx[icol]
            ax.set_title('{}={} | {}={}'.format(row, row_val, col, col_val))

    for ax in axes[:, 0]:
        ax.set_ylabel(y)

    for ax in axes[-1]:
        ax.set_xlabel(x)

    if legend_ax_idx:
        leg_lines = [l[0] for l in lines[legend_ax_idx]]
        leg_labels = [l.get_label() for l in leg_lines]
        axes[legend_ax_idx].legend(leg_lines, leg_labels, loc='best')

    if title:
        fig.suptitle(title)
    if tight_layout:
        fig.tight_layout()
    return SimpleNamespace(
        lines=lines,
        ax=ax,
        fig=fig,
    )


def cli_plot_gridified_truth(logpath, title_params, **kwargs):
    from ..loaders import load_records
    rec = load_records(logpath)
    kwargs.setdefault('title', rec.pretty_spec(title_params))
    arts = plot_gridified_truth(rec.truth_df(), **kwargs)
    return arts.fig


def main(args=None):
    from .basecli import make_base_parser, call_cli
    parser = make_base_parser(description=__doc__)
    add_arguments_from_function(parser, plot_gridified_truth)
    call_cli(cli_plot_gridified_truth, parser.parse_args(args))


if __name__ == '__main__':
    main()
