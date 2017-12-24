import numpy as np
import pandas
import pytest

from ...execution import DataStore
from ..datastore_loader import DataStoreLoader1

test_data_list = [
    np.array(
        [(0.0,),
         (1.0,)],
        dtype=[
            ('f0', 'double'),
        ]),
    np.array(
        [(0, 0.0, False),
         (1, 1.0, True)],
        dtype=[
            ('f0', 'uint32'),
            ('f1', 'double'),
            ('f2', 'b'),
        ]),
]


def assert_stored_data(directory, tablename, data, **kwargs):
    loader = DataStoreLoader1(directory)
    actual = loader.load(tablename)
    desired = pandas.DataFrame.from_records(data)
    pandas.testing.assert_frame_equal(actual, desired, **kwargs)


@pytest.mark.parametrize('data', test_data_list)
def test_load_csv(data, tmpdir):
    datastore_path = tmpdir.join('results').ensure(dir=True)
    tablename = 'data'
    filename = tablename + '.csv'

    with DataStore(str(datastore_path)) as datastore:
        tables = datastore.tables
        tables.saverow(filename, list(data.dtype.names))
        for row in data:
            tables.saverow(filename, list(row))

    assert_stored_data(str(datastore_path), tablename, data,
                       check_dtype=False)


@pytest.mark.parametrize('data', test_data_list)
def test_load_shared_hdf5(data, tmpdir):
    datastore_path = tmpdir.join('results').ensure(dir=True)
    tablename = 'data'

    with DataStore(str(datastore_path)) as datastore:
        tables = datastore.h5.tables
        for row in data:
            tables.saverow(tablename, row)

    assert_stored_data(str(datastore_path), tablename, data)
