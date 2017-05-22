from ..ssnode import sample_slice


def subsample_neurons(rate_vector, N, NZ, NB, sample_sites,
                      track_net_identity):
    """
    Reduce `rate_vector` into a form that can be fed to the discriminator.

    It works for both theano and numpy arrays.

    Parameters
    ----------
    rate_vector : array of shape (NZ, NB, 2N)
        Output of the SSN.
    track_net_identity : bool
        If False, squash all neurons into NZ axis; i.e., forget from
        which network the neurons are sampled.  If True, stack samples
        into NB axis; i.e., let discriminator know that those neurons
        are from the same SSN.

    Examples
    --------
    >>> import numpy as np
    >>> N, NZ, NB = 7, 5, 2
    >>> sample_sites = 3
    >>> rate_vector = np.tile(np.arange(2 * N), (NZ, NB, 1))
    >>> assert rate_vector.shape == (NZ, NB, 2 * N)
    >>> red0 = subsample_neurons(rate_vector, N, NZ, NB, sample_sites, False)
    >>> assert red0.shape == (NZ * sample_sites, NB)
    >>> red0
    array([[2, 2],
           [3, 3],
           [4, 4],
           [2, 2],
           [3, 3],
           [4, 4],
           [2, 2],
           [3, 3],
           [4, 4],
           [2, 2],
           [3, 3],
           [4, 4],
           [2, 2],
           [3, 3],
           [4, 4]])
    >>> red1 = subsample_neurons(rate_vector, N, NZ, NB, sample_sites, True)
    >>> assert red1.shape == (NZ, NB * sample_sites)
    >>> red1
    array([[2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4]])

    """
    probe = sample_slice(N, sample_sites)
    subsample = rate_vector[:, :, probe]
    if track_net_identity:
        return subsample.reshape((NZ, -1))
    else:
        return subsample.swapaxes(1, 2).reshape((-1, NB))
