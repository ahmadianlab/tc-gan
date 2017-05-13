from __future__ import print_function, division

from matplotlib import pyplot
import numpy as np

from .ssnode import sample_fixed_points


def plot_tunings(sample_sites=3,
                 bandwidths=[0, 0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1],
                 linewidth=0.2, ylim=(None, None),
                 **sample_kwargs):
    sample_kwargs.update(bandwidths=bandwidths)
    _Zs, rates, _info = sample_fixed_points(**sample_kwargs)
    rates = np.array(rates)
    N = rates.shape[-1] // 2
    i_beg = N // 2 - sample_sites // 3
    i_end = i_beg + sample_sites + 1
    tunings = rates[:, :, i_beg:i_end].swapaxes(0, 1)
    tunings = tunings.reshape((tunings.shape[0], -1))

    fig, ax = pyplot.subplots()
    ax.plot(bandwidths, tunings, color='C0', linewidth=linewidth)
    ax.set_xlabel('Bandwidths')
    ax.set_ylabel('Rates')
    ax.set_ylim(*ylim)

    return locals()
