from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas

from ...networks.utils import gridified_tc_axes
from ..records_loader import BaseRecords, MomentMatchingRecords


def fake_base_records():
    datasize = 11
    num_bandwidths = 7
    num_contrasts = 5
    num_probes = 3
    num_cell_types = 2
    shape = (datasize,
             num_contrasts,
             num_bandwidths,
             num_cell_types,
             num_probes)

    truth = np.arange(np.prod(shape)).reshape((-1, np.prod(shape[1:])))
    datastore = SimpleNamespace()
    datastore.load = mock.Mock(return_value=truth)

    rc = SimpleNamespace(
        datasize=datasize,
        bandwidths=list(map('b{}'.format, range(num_bandwidths))),
        contrasts=list(map('c{}'.format, range(num_contrasts))),
        norm_probes=list(map('p{}'.format, range(num_probes))),
        cell_types=list(map('t{}'.format, range(num_cell_types))),
    )

    return BaseRecords(datastore, info={}, rc=rc)


def test_truth_grid_vs_df():
    rec = fake_base_records()
    grid = rec.truth_grid()
    df = rec.truth_df()
    grid_df = df.reorder_levels(gridified_tc_axes[1:], axis='columns')

    # Sort columns of grid_df:
    columns = pandas.MultiIndex.from_product(grid_df.columns.levels)
    grid_df = grid_df.loc[:, columns]

    via_df = grid_df.as_matrix().reshape(grid.shape)
    np.testing.assert_equal(via_df, grid)


def test_pretty_spec_base():
    class RC(dict):
        gan_type = 'WGAN'

    rec = BaseRecords(
        datastore=None,
        info=None,
        rc=RC(alpha=1, beta=2),
    )
    actual = rec.pretty_spec(['alpha', 'beta'])
    assert actual == 'WGAN: alpha=1 beta=2'


def test_pretty_spec_moment_matching():
    rec = MomentMatchingRecords(
        datastore=None,
        info=None,
        rc={'alpha': 1, 'beta': 2},
    )
    actual = rec.pretty_spec(['alpha', 'beta'])
    assert actual == 'MM: alpha=1 beta=2'
