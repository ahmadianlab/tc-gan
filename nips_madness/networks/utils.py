def gridify_tc_data(
        data, num_contrasts, num_bandwidths, num_cell_types, num_probes):
    """
    Convert tuning curve `data` to the "grid" representation.

    .. parsed-literal::

      ,----- order of :term:`experiment parameter` varied in `data` given
      |  ,-- order of :term:`experiment parameter` varied in `grid` returned
      |  |
      0  0  ::  (samples / noise)
      1  3  ::  contrasts
      2  4  ::  bandwidths
      3  1  ::  cell types
      4  2  ::  probes

    Parameters
    ----------
    data : numpy.ndarray
        Tuning curve data with the constraint
        ``data.size == len(data) * "all num_* multiplied"``.
    num_bandwidths : int
    num_contrasts : int
    num_probes : int
    num_cell_types : int

    Returns
    -------
    grid : numpy.ndarray
        `data` reshaped and shuffled.

    Examples
    --------
    .. preparation
       >>> import numpy as np
       >>> num_bandwidths = 7
       >>> num_contrasts = 5
       >>> num_probes = 3
       >>> num_cell_types = 2
       >>> shape = (11, num_contrasts, num_bandwidths,
       ...          num_cell_types, num_probes)
       >>> data = np.arange(np.prod(shape)).reshape(shape)

    >>> grid = gridify_tc_data(
    ...    data, num_bandwidths=num_bandwidths, num_contrasts=num_contrasts,
    ...    num_probes=num_probes, num_cell_types=num_cell_types)
    >>> grid.shape == (len(data),
    ...                num_cell_types,
    ...                num_probes,
    ...                num_contrasts,
    ...                num_bandwidths)
    True

    """
    # This is the order of dimensions varied in subsample_neurons:
    grid = data.reshape((len(data), num_contrasts, num_bandwidths,
                         num_cell_types, num_probes))
    return grid.transpose((0, 3, 4, 1, 2))
