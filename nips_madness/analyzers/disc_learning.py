import json
import os

from matplotlib import pyplot

from ..utils import csv_line

defaults = dict(
    title_params=['layers', 'disc_normalization'],
    figsize=(9, 3),
)


class DiscriminatorLog(object):

    def __init__(self, logpath):
        import pandas

        if not os.path.isdir(logpath):
            logpath = os.path.dirname(logpath)
        self._datadir = logpath

        self.learning = pandas.read_csv(
            os.path.join(logpath, 'disc_learning.csv'))
        self.param_stats = pandas.read_csv(
            os.path.join(logpath, 'disc_param_stats.csv'))

        assert list(self.param_stats.columns[:2]) == ['gen_step', 'disc_step']
        self.param_stats_names = list(self.param_stats.columns[2:])

        self.learning['disc_updates'] = range(len(self.learning))
        self.param_stats['disc_updates'] = range(len(self.param_stats))

    def plot_learning(self, **kwargs):
        losses = ['Dloss', 'Daccuracy']
        self.learning.plot('disc_updates', losses, **kwargs)

    def plot_param_stats(self, logy=True, **kwargs):
        self.param_stats.plot('disc_updates', self.param_stats_names,
                              logy=logy,
                              **kwargs)

    def plot_all(self, title_params=defaults['title_params'],
                 figsize=defaults['figsize']):
        fig, axes = pyplot.subplots(ncols=2, sharex=True, squeeze=False,
                                    figsize=figsize)
        self.plot_learning(ax=axes[0, 0])
        self.plot_param_stats(ax=axes[0, 1])
        fig.suptitle(self.pretty_title(title_params))
        return fig

    def get_info(self):
        with open(os.path.join(self._datadir, 'info.json')) as file:
            return json.load(file)

    def pretty_title(self, title_params):
        run_config = self.get_info()['run_config']
        return ' '.join('{}={}'.format(k, run_config[k]) for k in title_params)


load_disc_log = DiscriminatorLog


def analyze_disc_learning(logpath, figpath=None, show=False, **kwds):
    """
    Plot learning curves of a discriminator loaded from `logpath`.
    """
    disc = load_disc_log(logpath)
    fig = disc.plot_all(**kwds)
    if show:
        pyplot.show()
    if figpath:
        fig.savefig(figpath)


def main(args=None):
    import argparse

    class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=analyze_disc_learning.__doc__)
    parser.add_argument('logpath')
    parser.add_argument(
        '--figsize', type=csv_line(float),
        help='Width and height (in inch) separated by a comma',
        default=defaults['figsize'])
    parser.add_argument('--figpath')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--title-params', type=csv_line(str),
        default=defaults['title_params'],
        help='Comma separated name of parameters to be used in figure title.')
    ns = parser.parse_args(args)
    analyze_disc_learning(**vars(ns))


if __name__ == '__main__':
    main()
