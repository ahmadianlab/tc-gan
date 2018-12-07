from matplotlib import pyplot

from .loader import load_gandata
from .learning import plot_tuning_curve_evo


def run_axplotter(plotter, logpath, figpath, show, **kwargs):
    fig, ax = pyplot.subplots()
    data = load_gandata(logpath)
    plotter(data, ax=ax, **kwargs)
    ax.set_title(data.pretty_spec())
    if show:
        pyplot.show()
    if figpath:
        fig.savefig(figpath)


def cli_axplotter(plotter, args=None, **kwds):
    def executor(setup_parser):
        import argparse
        parser = argparse.ArgumentParser(
            formatter_class=type('FormatterClass',
                                 (argparse.RawDescriptionHelpFormatter,
                                  argparse.ArgumentDefaultsHelpFormatter),
                                 {}),
            **kwds)
        parser.add_argument('--figpath')
        parser.add_argument('--show', action='store_true')
        parser.add_argument('logpath', help='path to SSNGAN_log_*.log')
        setup_parser(parser)
        ns = parser.parse_args(args)
        run_axplotter(plotter, **vars(ns))
    return executor


def main(args=None):
    @cli_axplotter(plot_tuning_curve_evo, args)
    def _(parser):
        parser.add_argument('--epochs', type=eval)


if __name__ == '__main__':
    main()
