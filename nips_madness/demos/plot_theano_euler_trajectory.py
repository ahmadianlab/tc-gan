from matplotlib import pyplot

from ..utils import log_timing
from ..networks.fixed_time_sampler import FixedTimeTuningCurveSampler


def plot_theano_euler_trajectory(**sampler_config):
    sampler_config['batchsize'] = 1
    sampler = FixedTimeTuningCurveSampler.from_dict(sampler_config)
    sampler.prepare()
    with log_timing("sampler.compute_trajectories()"):
        trajectories = sampler.compute_trajectories()
    trajectories.plot()
    pyplot.show()


def main(args=None):
    import argparse
    from ..networks.fixed_time_sampler import add_arguments

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)

    add_arguments(parser, exclude=['batchsize'])

    ns = parser.parse_args(args)
    plot_theano_euler_trajectory(**vars(ns))


if __name__ == '__main__':
    main()
