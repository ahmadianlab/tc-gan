from matplotlib import pyplot

from ..utils import log_timing
from ..networks.fixed_time_sampler import FixedTimeTuningCurveSampler


def plot_tunings(tc, sampler):
    fig, ax = pyplot.subplots()
    ax.plot(tc.T)


def plot_theano_euler_tunings(**sampler_config):
    sampler = FixedTimeTuningCurveSampler.from_dict(sampler_config)
    sampler.prepare()
    with log_timing("sampler.forward()"):
        out = sampler.forward(full_output=True)
    tc = out.prober_tuning_curve
    plot_tunings(tc, sampler)
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
    add_arguments(parser)
    parser.set_defaults(batchsize=10)
    ns = parser.parse_args(args)
    plot_theano_euler_tunings(**vars(ns))


if __name__ == '__main__':
    main()
