import logging
import os
import pickle
import sys
from unittest import mock
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litserve.utils import (
    call_after_stream,
    configure_logging,
    dump_exception,
    generate_random_zmq_address,
    is_package_installed,
    set_trace_if_debug,
)


def test_dump_exception():
    e1 = dump_exception(HTTPException(status_code=404, detail="Not Found"))
    assert isinstance(e1, bytes)

    exc = HTTPException(400, "Custom Lit error")
    isinstance(pickle.loads(dump_exception(exc)), HTTPException)
    assert pickle.loads(dump_exception(exc)).detail == "Custom Lit error"
    assert pickle.loads(dump_exception(exc)).status_code == 400


async def dummy_streamer():
    for i in range(10):
        yield i


@pytest.mark.asyncio
async def test_call_after_stream():
    callback = MagicMock()
    callback.return_value = None
    streamer = dummy_streamer()
    async for _ in call_after_stream(streamer, callback, "first_arg", random_arg="second_arg"):
        pass
    callback.assert_called()
    callback.assert_called_with("first_arg", random_arg="second_arg")


@pytest.mark.skipif(sys.platform == "win32", reason="This test is for non-Windows platforms only.")
def test_generate_random_zmq_address_non_windows(tmpdir):
    """Test generate_random_zmq_address on non-Windows platforms."""

    temp_dir = str(tmpdir)
    address1 = generate_random_zmq_address(temp_dir=temp_dir)
    address2 = generate_random_zmq_address(temp_dir=temp_dir)

    assert address1.startswith("ipc://"), "Address should start with 'ipc://'"
    assert address2.startswith("ipc://"), "Address should start with 'ipc://'"
    assert address1 != address2, "Addresses should be unique"

    # Verify the path exists within the specified temp_dir
    assert os.path.commonpath([temp_dir, address1[6:]]) == temp_dir
    assert os.path.commonpath([temp_dir, address2[6:]]) == temp_dir


def test_configure_logging():
    configure_logging(use_rich=False)
    assert logging.getLogger("litserve").handlers[0].__class__.__name__ == "StreamHandler"


def test_configure_logging_rich_not_installed():
    # patch builtins.__import__ to raise ImportError
    with mock.patch("builtins.__import__", side_effect=ImportError):
        configure_logging(use_rich=True)
        assert logging.getLogger("litserve").handlers[0].__class__.__name__ == "StreamHandler"


@mock.patch("litserve.utils.set_trace")
def test_set_trace_if_debug(mock_set_trace):
    # mock environ
    with mock.patch("litserve.utils.os.environ", {"LITSERVE_DEBUG": "1"}):
        set_trace_if_debug()
    mock_set_trace.assert_called_once()


@mock.patch("litserve.utils.ForkedPdb")
def test_set_trace_if_debug_not_set(mock_forked_pdb):
    with mock.patch("litserve.utils.os.environ", {"LITSERVE_DEBUG": "0"}):
        set_trace_if_debug()
    mock_forked_pdb.assert_not_called()


def test_is_package_installed():
    assert is_package_installed("pytest")
    assert not is_package_installed("nonexistent_package")
