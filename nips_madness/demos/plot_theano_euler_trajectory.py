from matplotlib import pyplot

from ..networks.finite_time_sampler import FiniteTimeTuningCurveSampler


def plot_trajectory(trajectories, sampler):
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

    axes[0, 0].set_title('Excitatory neurons')
    axes[0, 1].set_title('Inhibitory neurons')

    for ax in axes[-1]:
        ax.set_xlabel('Time')

    for ax in axes[:, 0]:
        ax.set_ylabel('Rate')


def plot_theano_euler_trajectory(**sampler_config):
    sampler = FiniteTimeTuningCurveSampler.from_dict(sampler_config)
    trajectories = sampler.compute_trajectories()
    plot_trajectory(trajectories, sampler)
    pyplot.show()


def main(args=None):
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)
    ns = parser.parse_args(args)
    plot_theano_euler_trajectory(**vars(ns))


if __name__ == '__main__':
    main()
