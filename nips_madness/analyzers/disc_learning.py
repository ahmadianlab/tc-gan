import json
import os
import warnings

from matplotlib import pyplot
import numpy as np

from ..utils import csv_line

defaults = dict(
    title_params=['layers', 'disc_normalization'],
    figsize=(9, 3),
)


def param_stats_to_pretty_label(name):
    """
    Convert param_stats names to latex label for plotting.

    >>> param_stats_to_pretty_label('W.nnorm')
    '$|W_0|$'
    >>> param_stats_to_pretty_label('W.nnorm.1')
    '$|W_1|$'
    >>> param_stats_to_pretty_label('b.nnorm.2')
    '$|b_2|$'
    >>> param_stats_to_pretty_label('scales.nnorm.3')
    '$|g_3|$'
    >>> param_stats_to_pretty_label('spam')

    """
    parts = name.split('.')
    if 2 <= len(parts) <= 3 and parts[1] == 'nnorm':
        var = parts[0]
        suffix = parts[2] if len(parts) == 3 else '0'
        if var == 'scales':
            var = 'g'
            # "g" for gain; see: Ba et al (2016) Layer Normalization
        return '$|{}_{}|$'.format(var, suffix)


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

        rows_learning = len(self.learning)
        rows_param_stats = len(self.param_stats)
        if rows_learning != rows_param_stats:
            min_rows = min(rows_learning, rows_param_stats)
            self.learning = self.learning.iloc[:, :min_rows]
            self.param_stats = self.param_stats.iloc[:, :min_rows]

            warnings.warn(
                'Rows in disc_learning.csv ({}) and rows in'
                ' disc_param_stats.csv ({}) differ.'
                ' Resetting the shortest ({}).'
                .format(rows_learning, rows_param_stats, min_rows))

        assert list(self.param_stats.columns[:2]) == ['gen_step', 'disc_step']
        self.param_stats_names = list(self.param_stats.columns[2:])

        self.learning['disc_updates'] = range(len(self.learning))
        self.param_stats['disc_updates'] = range(len(self.param_stats))
        self.learning['epoch'] = self.param_stats['epoch'] = self.epochs

    @property
    def disc_updates(self):
        return np.asarray(self.learning['disc_updates'])

    @property
    def epochs(self):
        return self.disc_updates_to_epoch(self.disc_updates)

    def disc_updates_to_epoch(self, disc_updates):
        run_config = self.get_info()['run_config']
        truth_size = run_config['truth_size']  # data size
        n_samples = self.batchsize
        return disc_updates * n_samples / truth_size

    @property
    def batchsize(self):
        run_config = self.get_info()['run_config']
        try:
            return run_config['batchsize']
        except KeyError:
            return run_config['n_samples']

    def plot_learning(self, yscale='symlog', **kwargs):
        losses = ['Dloss', 'Daccuracy']
        ax = self.learning.plot('epoch', losses, **kwargs)
        ax.set_yscale(yscale)
        return ax

    def plot_param_stats(
            self, logy=True,
            legend=dict(loc='center left', ncol='auto', fontsize='small',
                        handlelength=0.5, columnspacing=0.4),
            legend_max_rows=7,
            **kwargs):
        ax = self.param_stats.plot('epoch', self.param_stats_names,
                                   logy=logy,
                                   legend=False,
                                   **kwargs)
        for line in ax.get_lines():
            label = param_stats_to_pretty_label(line.get_label())
            if label:
                line.set_label(label)
        if legend:
            if not isinstance(legend, dict):
                legend = {}
            legend = dict(legend)
            if legend.get('ncol') == 'auto':
                n_stats = len(self.param_stats_names)
                legend['ncol'] = int(np.ceil(n_stats / legend_max_rows))
            leg = ax.legend(**legend)
            leg.set_frame_on(True)
        return ax

    def plot_all(self, title_params=defaults['title_params'],
                 figsize=defaults['figsize']):
        fig, axes = pyplot.subplots(ncols=2, sharex=True, squeeze=False,
                                    figsize=figsize)
        self.plot_learning(ax=axes[0, 0])
        axes[0, 0].axhline(0, color='0.5')
        self.plot_param_stats(ax=axes[0, 1])
        fig.suptitle(self.pretty_title(title_params))
        return fig

    def get_info(self):
        with open(os.path.join(self._datadir, 'info.json')) as file:
            return json.load(file)

    def pretty_title(self, title_params):
        info = self.get_info()
        run_config = dict(info['extra_info'], **info['run_config'])
        run_config['n_bandwidths'] = len(run_config['bandwidths'])
        return ' '.join('{}={}'.format(k, run_config[k]) for k in title_params)


load_disc_log = DiscriminatorLog


def analyze_disc_learning(logpath, figpath=None, show=False, **kwds):
    """
    Plot learning curves of a discriminator loaded from `logpath`.
    """
    disc = load_disc_log(logpath)
    fig = disc.plot_all(**kwds)
    fig.tight_layout()
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
