import numpy as np


def sample_slice(N, center_sites):
    """
    Generate a slice for sampling `center_sites` from an array of `N` neurons.

    >>> N = 201
    >>> neurons = np.arange(N, dtype=int)
    >>> neurons[sample_slice(N, 3)]
    array([ 99, 100, 101])
    >>> neurons[sample_slice(N, 4)]
    array([ 98,  99, 100, 101])
    >>> neurons[sample_slice(N, 5)]
    array([ 98,  99, 100, 101, 102])

    """
    i_beg = N // 2 - center_sites // 2
    i_end = i_beg + center_sites
    return np.s_[i_beg:i_end]


def sample_sites_from_stim_space(stim_locs, N):
    """
    Make `sample_sites` from specification on stimulation space (bandwidth).

    Parameters
    ----------
    stim_locs : list of floats
        Locations in stimulation space [-1, 1].
    N : int
        Number of neurons.

    Examples
    --------
    >>> sample_sites_from_stim_space([0, 0.5, 1], 101)
    [50, 75, 100]

    This function assumes that `N` is large enough for given
    `stim_locs` so that generated indices are unique.  Otherwise,
    it raises an exception:

    >>> sample_sites_from_stim_space([0, 0.0001], 101)
    ...                                    # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    ValueError: Non-unique sample sites are specified.
    N (= 101) is not large enough for stim_locs (= [0.0, 0.0001])
    to generate unique sample sites.
    They generates sample_sites = [50, 50]

    """
    stim_locs = np.asarray(stim_locs)
    assert all(stim_locs >= -1)
    assert all(stim_locs <= 1)

    sample_sites = ((stim_locs + 1) * (N - 1) / 2).astype(int)

    if len(sample_sites) != len(set(sample_sites)):
        raise ValueError(
            'Non-unique sample sites are specified.\n'
            'N (= {}) is not large enough for stim_locs (= {}) to'
            ' generate unique sample sites.'
            ' They generates sample_sites = {}'
            .format(N, list(stim_locs), list(sample_sites)))

    return list(sample_sites)


def subsample_neurons(rate_vector, sample_sites,
                      track_offset_identity=False,
                      include_inhibitory_neurons=False,
                      N=None, NZ=None, NB=None):
    """
    Reduce `rate_vector` into a form that can be fed to the discriminator.

    It works for both theano and numpy arrays.

    Parameters
    ----------
    rate_vector : array of shape (NZ, NB, 2N)
        Output of the SSN.
    sample_sites : list of ints
        Probe locations in neural index (0 to N - 1).
    track_offset_identity : bool
        If False, squash all neurons into NZ axis; i.e., forget from
        which probe offset the neurons are sampled.  If True, stack samples
        into NB axis; i.e., let discriminator know that those neurons
        are from the different offset of the same SSN.

    Examples
    --------
    >>> N, NZ, NB = 7, 5, 2
    >>> sample_sites = [2, 3, 4]
    >>> rate_vector = np.tile(np.arange(2 * N), (NZ, NB, 1))
    >>> assert rate_vector.shape == (NZ, NB, 2 * N)
    >>> red0 = subsample_neurons(rate_vector, sample_sites, False)
    >>> assert red0.shape == (NZ * len(sample_sites), NB)
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
    >>> assert red1.shape == (NZ, NB * len(sample_sites))
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
        assert 0 <= min(sample_sites)
        assert max(sample_sites) < N

    if include_inhibitory_neurons:
        sample_sites = list(sample_sites)  # copy
        sample_sites.extend(np.array(sample_sites) + N)

    subsample = rate_vector[:, :, sample_sites]
    if track_offset_identity:
        return subsample.reshape((NZ, -1))
    else:
        return subsample.swapaxes(1, 2).reshape((-1, NB))
