from matplotlib import pyplot

from .loader import load_gandata


def plot_learning(data):
    df = data.to_dataframe()
    fig, axes = pyplot.subplots(nrows=2, ncols=2,
                                squeeze=False, figsize=(9, 6))
    df.plot('epoch', ['Gloss', 'Dloss'], ax=axes[0, 0])
    df.plot('epoch', ['Daccuracy'], ax=axes[0, 1])
    df.plot('epoch', ['SSsolve_time', 'gradient_time'], ax=axes[1, 0])
    df.plot('epoch', ['model_convergence', 'truth_convergence'], ax=axes[1, 1])
    fig.suptitle(data.tag)
    return fig


def analyze_learning(logpath, show, figpath):
    fig = plot_learning(load_gandata(logpath))
    if show:
        pyplot.show()
    if figpath:
        fig.savefig(figpath)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logpath', help='path to SSNGAN_log_*.log')
    parser.add_argument('--figpath')
    parser.add_argument('--show', action='store_true')
    ns = parser.parse_args(args)
    analyze_learning(**vars(ns))


if __name__ == '__main__':
    main()
