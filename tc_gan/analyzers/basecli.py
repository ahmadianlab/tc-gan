import argparse

from ..utils import csv_line, log_timing


class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass


def add_base_arguments(parser):
    parser.add_argument(
        'logpath',
        help='''Path to GAN output directory. It can also be any file
        in such directory; filename part is ignored.''')
    parser.add_argument(
        '--figpath',
        help='path to save figure')
    parser.add_argument(
        '--show', action='store_true',
        help='popup figure window')
    parser.add_argument(
        '--title-params', type=csv_line(str),
        help='Comma separated name of parameters to be used in figure title.')


def make_base_parser(formatter_class=CustomFormatter, **kwargs):
    parser = argparse.ArgumentParser(formatter_class=formatter_class, **kwargs)
    add_base_arguments(parser)
    return parser


def call_cli(func, ns):
    def wrapper(figpath, show, **kwargs):
        from matplotlib import pyplot
        fig = func(**kwargs)
        if show:
            pyplot.show()
        if figpath:
            with log_timing('fig.savefig()'):
                fig.savefig(figpath)
    return wrapper(**vars(ns))
