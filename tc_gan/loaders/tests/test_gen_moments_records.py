from types import SimpleNamespace

import numpy as np
import pandas
import pytest

from ...execution import DataStore
from ...recorders import GenMomentsRecorder
from ..datastore_loader import DataStoreLoader1


@pytest.mark.parametrize('num_mom_conds', [1, 2, 12])
def test_record_load(num_mom_conds, tmpdir):
    datastore = DataStore(str(tmpdir))
    recorder = GenMomentsRecorder(datastore, num_mom_conds)

    num_steps = 10
    mom_shape = (num_steps, 2 * num_mom_conds)
    desired = pandas.DataFrame(
        np.arange(np.prod(mom_shape)).reshape(mom_shape),
        columns=pandas.MultiIndex.from_product([['mean', 'var'],
                                                range(num_mom_conds)]),
        dtype='double')
    desired['step'] = np.arange(num_steps, dtype='uint32')

    for gen_step in range(num_steps):
        update_result = SimpleNamespace(gen_moments=np.asarray([
            desired.loc[gen_step, 'mean'],
            desired.loc[gen_step, 'var'],
        ]))
        recorder.record(gen_step, update_result)

    loader = DataStoreLoader1(str(tmpdir))
    actual = loader.load('gen_moments')
    pandas.testing.assert_frame_equal(actual, desired)
