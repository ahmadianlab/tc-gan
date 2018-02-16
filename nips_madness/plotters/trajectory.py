from matplotlib import pyplot


def plot_trajectory(trajectories, sampler, i_batch=0):
    trajectories = trajectories[i_batch]
    nrows = len(trajectories)
    fig, axes = pyplot.subplots(
        nrows=nrows,
        ncols=2,
        squeeze=False,
        figsize=(6, nrows * 1.5),
    )

    num_sites = sampler.num_sites
    ts = sampler.timepoints()
    for (ax_E, ax_I), rate in zip(axes, trajectories):
        ax_E.plot(ts, rate[:, :num_sites], color='C0', linewidth=0.5)
        ax_I.plot(ts, rate[:, num_sites:], color='C1', linewidth=0.5)

    for ax, point in zip(axes[:, 0], sampler.dom_points):
        text = '\n'.join(map('{0[0]}={0[1]}'.format, point.items()))
        ax.text(0.02, 0.85, text, transform=ax.transAxes,
                fontsize='medium', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axes[0, 0].set_title('Excitatory neurons')
    axes[0, 1].set_title('Inhibitory neurons')

    for ax in axes[-1]:
        ax.set_xlabel('Time')

    for ax in axes[:, 0]:
        ax.set_ylabel('Rate')

    return fig
