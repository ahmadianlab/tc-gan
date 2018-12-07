from unittest import mock
import io

import numpy
import pytest

from ..execution import DataTables, run_exit_hooks


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


def test_run_exit_hooks():
    class CustomException(Exception):
        pass

    def raising_exitter():
        hook = mock.Mock()
        hook.side_effect = CustomException
        return hook

    exit_hooks = [
        raising_exitter(),
        raising_exitter(),
        raising_exitter(),
        raising_exitter(),
    ]

    with pytest.raises(CustomException):
        run_exit_hooks(exit_hooks)

    for hook in exit_hooks:
        hook.assert_called_once()
