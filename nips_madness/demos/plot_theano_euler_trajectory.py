from matplotlib import pyplot

from ..utils import log_timing
from ..networks.finite_time_sampler import FiniteTimeTuningCurveSampler
from ..networks.ssn import ssn_impl_choices, ssn_type_choices


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


def plot_theano_euler_trajectory(**sampler_config):
    sampler_config['batchsize'] = 1
    sampler = FiniteTimeTuningCurveSampler.from_dict(sampler_config)
    sampler.prepare()
    with log_timing("sampler.compute_trajectories()"):
        trajectories, = sampler.compute_trajectories()
    plot_trajectory(trajectories, sampler)
    pyplot.show()


def main(args=None):
    import argparse
    from ..networks.finite_time_sampler import DEFAULT_PARAMS

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)

    parser.add_argument(
        '--ssn-impl', default=ssn_impl_choices[0], choices=ssn_impl_choices,
        help="SSN implementation.")
    parser.add_argument(
        '--ssn-type', default=ssn_type_choices[0], choices=ssn_type_choices,
        help="SSN type.")

    for key in sorted(DEFAULT_PARAMS):
        val = DEFAULT_PARAMS[key]
        if isinstance(val, (str, float, int)):
            argtype = type(val)
        else:
            argtype = eval
        parser.add_argument(
            '--{}'.format(key), type=argtype, default=val,
            help='for SSN')

    ns = parser.parse_args(args)
    plot_theano_euler_trajectory(**vars(ns))


if __name__ == '__main__':
    main()
