from unittest import mock

import pytest

from ..execution import KnownError
from ..drivers import recording_exit_reason


def test_record_exit_upon_end_of_iteration():
    datastore = mock.Mock()
    with recording_exit_reason(datastore):
        pass

    datastore.save_exit_reason.assert_called_once_with(
        reason='end_of_iteration',
        good=True,
    )


def test_record_exit_upon_keyboard_interrupt():
    datastore = mock.Mock()
    with pytest.raises(KeyboardInterrupt):
        with recording_exit_reason(datastore):
            raise KeyboardInterrupt

    datastore.save_exit_reason.assert_called_once_with(
        reason='keyboard_interrupt',
        good=False,
    )


def test_record_exit_upon_uncaught_exception():
    datastore = mock.Mock()
    with pytest.raises(Exception):
        with recording_exit_reason(datastore):
            exception = Exception('some exception')
            raise exception

    datastore.save_exit_reason.assert_called_once_with(
        reason='uncaught_exception',
        good=False,
        exception=str(exception),
    )


def test_record_exit_upon_known_error():
    datastore = mock.Mock()
    with pytest.raises(KnownError):
        with recording_exit_reason(datastore):
            raise KnownError('message')

    datastore.save_exit_reason.assert_not_called()
