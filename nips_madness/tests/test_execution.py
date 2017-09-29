import io

import numpy

from ..execution import DataTables


def test_datatables():
    tables = DataTables(None)
    tables._open = lambda _: io.StringIO()

    with tables:
        tables.saverow('A', [1, 2, 3])
        tables.saverow('A', [4, 5, 6])

        sio = tables._files['A']
        buf = io.BytesIO(sio.getvalue().encode())

        assert not sio.closed
    assert sio.closed

    data = numpy.loadtxt(buf, delimiter=',')
    numpy.testing.assert_equal(data, [[1, 2, 3], [4, 5, 6]])
