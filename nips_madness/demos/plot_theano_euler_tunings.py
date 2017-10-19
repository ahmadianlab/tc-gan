from matplotlib import pyplot

from ..networks.finite_time_sampler import FiniteTimeTuningCurveSampler


def plot_tunings(tc, sampler):
    fig, ax = pyplot.subplots()
    ax.plot(tc.T)


def plot_theano_euler_tunings(**sampler_config):
    sampler = FiniteTimeTuningCurveSampler.from_dict(sampler_config)
    out = sampler.forward(full_output=True)
    tc = out.prober_tuning_curve
    plot_tunings(tc, sampler)
    pyplot.show()


def main(args=None):
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)
    parser.add_argument('--batchsize', default=10, type=int)
    ns = parser.parse_args(args)
    plot_theano_euler_tunings(**vars(ns))


if __name__ == '__main__':
    main()
