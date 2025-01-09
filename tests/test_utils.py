import pickle
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litserve.utils import call_after_stream, dump_exception, get_random_port


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


def test_get_random_port():
    port = get_random_port()
    assert 1024 <= port <= 65535
    assert isinstance(port, int)
