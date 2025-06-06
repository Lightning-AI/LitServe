import asyncio
import inspect
import logging
import os
import pickle
import sys
from unittest import mock
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litserve.utils import (
    asyncify,
    call_after_stream,
    configure_logging,
    dump_exception,
    generate_random_zmq_address,
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


# Tests for asyncify function
def sync_function(x):
    """Test sync function."""
    return x * 2


def sync_generator(n):
    """Test sync generator."""
    yield from range(n)


async def async_function(x):
    """Test async function."""
    return x * 2


async def async_generator(n):
    """Test async generator."""
    for i in range(n):
        yield i


@pytest.mark.asyncio
async def test_asyncify_sync_function():
    """Test asyncify with regular sync function."""
    async_func = asyncify(sync_function)
    assert asyncio.iscoroutinefunction(async_func)
    result = await async_func(5)
    assert result == 10


@pytest.mark.asyncio
async def test_asyncify_sync_generator():
    """Test asyncify with sync generator."""
    async_gen = asyncify(sync_generator)
    assert inspect.isasyncgenfunction(async_gen)
    results = []
    async for value in async_gen(3):
        results.append(value)
    assert results == [0, 1, 2]


@pytest.mark.asyncio
async def test_asyncify_async_function_passthrough():
    """Test asyncify preserves async functions as-is."""
    async_func = asyncify(async_function)
    assert async_func is async_function  # Should be the same object
    result = await async_func(5)
    assert result == 10


@pytest.mark.asyncio
async def test_asyncify_async_generator_passthrough():
    """Test asyncify preserves async generators as-is."""
    async_gen = asyncify(async_generator)
    assert async_gen is async_generator  # Should be the same object
    results = []
    async for value in async_gen(3):
        results.append(value)
    assert results == [0, 1, 2]


@pytest.mark.asyncio
async def test_asyncify_preserves_function_metadata():
    """Test that asyncify preserves function metadata using functools.wraps."""
    async_func = asyncify(sync_function)
    assert async_func.__name__ == sync_function.__name__
    assert async_func.__doc__ == sync_function.__doc__


@pytest.mark.asyncio
async def test_asyncify_with_args_and_kwargs():
    """Test asyncify works with functions that take args and kwargs."""

    def sync_func_with_args(a, b, c=10, d=20):
        return a + b + c + d

    async_func = asyncify(sync_func_with_args)
    result = await async_func(1, 2, c=3, d=4)
    assert result == 10


@pytest.mark.asyncio
async def test_asyncify_sync_generator_with_exception():
    """Test asyncify handles exceptions in sync generators."""

    def failing_generator():
        yield 1
        yield 2
        raise ValueError("Test error")

    async_gen = asyncify(failing_generator)

    # Collect values until exception
    async def collect_values():
        results = []
        async for value in async_gen():
            results.append(value)
        return results

    with pytest.raises(ValueError, match="Test error"):
        await collect_values()


@pytest.mark.asyncio
async def test_asyncify_sync_function_with_exception():
    """Test asyncify handles exceptions in sync functions."""

    def failing_function():
        raise RuntimeError("Test sync error")

    async_func = asyncify(failing_function)

    with pytest.raises(RuntimeError, match="Test sync error"):
        await async_func()
