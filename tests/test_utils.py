import os
import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litserve.utils import call_after_stream, dump_exception, generate_random_zmq_address


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


@patch("sys.platform", "win32")
@patch("zmq.Context")
def test_generate_random_zmq_address_windows(mock_ctx):
    """Test generate_random_zmq_address on Windows platforms."""
    mock_socket = mock_ctx.return_value.socket.return_value
    mock_socket.bind_to_random_port.return_value = 5555

    address = generate_random_zmq_address()
    assert address == "tcp://localhost:5555"

    # Verify socket and context were properly used
    mock_ctx.return_value.socket.assert_called_once()
    mock_socket.bind_to_random_port.assert_called_once_with("localhost")
    mock_socket.close.assert_called_once()
    mock_ctx.return_value.term.assert_called_once()
