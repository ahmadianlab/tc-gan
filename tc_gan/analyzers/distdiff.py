from __future__ import print_function

from matplotlib import pyplot
import numpy as np
import scipy.stats

from ..utils import make_progressbar
from ..ssnode import sample_tuning_curves
from .loader import load_gandata


def generated_tuning_curves(data, indices=None, **kwargs):
    for J, D, S in data.iter_gen_params(indices=indices):
        yield sample_tuning_curves(J=J, D=D, S=S, **kwargs)


def calc_distdiff(data, true_io_type=None, sample_indices=None,
                  quiet=False, **kwargs):
    if true_io_type is None:
        true_io_type = data.io_type
    solver_kwargs = dict(io_type=true_io_type, **kwargs)
    truths, _ = sample_tuning_curves(**solver_kwargs)

    def alloc():
        return [[] for _ in range(len(truths))]

    if sample_indices is None:
        epochs = range(len(data.gen))
    else:
        epochs = range(len(data.gen))[sample_indices]

    results = dict(KSD=alloc(), KSp=alloc())
    bar = make_progressbar(quiet=quiet, max_value=len(epochs))
    for fake, _ in bar(generated_tuning_curves(data, indices=sample_indices,
                                               **kwargs)):
        for i, (t_samples, f_samples) in enumerate(zip(truths, fake)):
            d, p = scipy.stats.ks_2samp(t_samples, f_samples)
            results['KSD'][i].append(d)
            results['KSp'][i].append(p)

    results = {key: np.array(val) for key, val in results.items()}
    results['epochs'] = epochs
    return results


def plot_distdiff(results, **plot_kwds):
    fig, ax = pyplot.subplots()
    for i, KSD in enumerate(results['KSD']):
        ax.plot(results['epochs'], KSD, label=str(i), **plot_kwds)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Kolmogorov-Smirnov statistic')
    ax.set_yscale('log')
    ax.legend(loc='best')
    fig.tight_layout()
    return fig


def run_distdiff(logpath, NZ=30, show=False, figpath=None):
    data = load_gandata(logpath)
    results = calc_distdiff(data, NZ=NZ)
    fig = plot_distdiff(results)
    if show:
        pyplot.show()
    if figpath:
        fig.savefig(figpath)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logpath')
    parser.add_argument('--NZ', default=30, type=int)
    parser.add_argument('--figpath')
    parser.add_argument('--show', action='store_true')
    ns = parser.parse_args(args)
    run_distdiff(**vars(ns))


if __name__ == '__main__':
    main()
