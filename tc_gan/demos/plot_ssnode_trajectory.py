from matplotlib import pyplot

from ..ssnode import plot_trajectory, DEFAULT_PARAMS
from ..utils import add_arguments_from_dict


def main(args=None):
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=__doc__)
    parser.add_argument('--plot-fp', action='store_true',
                        help='Include fixed points in the plot.')
    add_arguments_from_dict(
        parser, DEFAULT_PARAMS,
        exclude=['bandwidths', 'offset', 'rate_soft_bound', 'rate_hard_bound',
                 'tau'])
    ns = parser.parse_args(args)

    plot_trajectory(**vars(ns))
    pyplot.show()


if __name__ == '__main__':
    main()
