import numpy as np


def sample_slice(N, sample_sites):
    """
    Generate a slice for sampling `sample_sites` from an array of `N` neurons.

    >>> N = 201
    >>> neurons = np.arange(N, dtype=int)
    >>> neurons[sample_slice(N, 3)]
    array([ 99, 100, 101])
    >>> neurons[sample_slice(N, 4)]
    array([ 98,  99, 100, 101])
    >>> neurons[sample_slice(N, 5)]
    array([ 98,  99, 100, 101, 102])

    """
    i_beg = N // 2 - sample_sites // 2
    i_end = i_beg + sample_sites
    return np.s_[i_beg:i_end]


def subsample_neurons(rate_vector, sample_sites,
                      track_offset_identity=False,
                      N=None, NZ=None, NB=None):
    """
    Reduce `rate_vector` into a form that can be fed to the discriminator.

    It works for both theano and numpy arrays.

    Parameters
    ----------
    rate_vector : array of shape (NZ, NB, 2N)
        Output of the SSN.
    track_offset_identity : bool
        If False, squash all neurons into NZ axis; i.e., forget from
        which probe offset the neurons are sampled.  If True, stack samples
        into NB axis; i.e., let discriminator know that those neurons
        are from the different offset of the same SSN.

    Examples
    --------
    >>> N, NZ, NB = 7, 5, 2
    >>> sample_sites = 3
    >>> rate_vector = np.tile(np.arange(2 * N), (NZ, NB, 1))
    >>> assert rate_vector.shape == (NZ, NB, 2 * N)
    >>> red0 = subsample_neurons(rate_vector, sample_sites, False)
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
    >>> red1 = subsample_neurons(rate_vector, sample_sites, True)
    >>> assert red1.shape == (NZ, NB * sample_sites)
    >>> red1
    array([[2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4],
           [2, 3, 4, 2, 3, 4]])

    """
    if isinstance(rate_vector, np.ndarray):
        NZ_, NB_, TN_ = rate_vector.shape
        if NZ is None:
            NZ = NZ_
        if NB is None:
            NB = NB_
        if N is None:
            N = TN_ // 2
        assert (NZ_, NB_, TN_) == (NZ, NB, 2 * N)

    probe = sample_slice(N, sample_sites)
    subsample = rate_vector[:, :, probe]
    if track_offset_identity:
        return subsample.reshape((NZ, -1))
    else:
        return subsample.swapaxes(1, 2).reshape((-1, NB))
