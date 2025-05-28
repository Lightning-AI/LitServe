# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import dataclasses
import inspect
import logging
import os
import pdb
import pickle
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import HTTPException

if TYPE_CHECKING:
    from litserve.server import LitServer

logger = logging.getLogger(__name__)


class LitAPIStatus:
    OK = "OK"
    ERROR = "ERROR"
    FINISH_STREAMING = "FINISH_STREAMING"


class PickleableHTTPException(HTTPException):
    @staticmethod
    def from_exception(exc: HTTPException):
        status_code = exc.status_code
        detail = exc.detail
        return PickleableHTTPException(status_code, detail)

    def __reduce__(self):
        return (HTTPException, (self.status_code, self.detail))


def dump_exception(exception):
    if isinstance(exception, HTTPException):
        exception = PickleableHTTPException.from_exception(exception)
    return pickle.dumps(exception)


async def azip(*async_iterables):
    iterators = [ait.__aiter__() for ait in async_iterables]
    while True:
        results = await asyncio.gather(*(ait.__anext__() for ait in iterators), return_exceptions=True)
        if any(isinstance(result, StopAsyncIteration) for result in results):
            break
        yield tuple(results)


@contextmanager
def wrap_litserve_start(server: "LitServer"):
    """Pytest utility to start the server in a context manager."""
    server.app.response_queue_id = 0
    for lit_api in server.litapi_connector:
        if lit_api.spec:
            lit_api.spec.response_queue_id = 0

    manager = server._init_manager(1)
    processes = []
    for lit_api in server.litapi_connector:
        processes.extend(server.launch_inference_worker(lit_api))
    server._prepare_app_run(server.app)
    try:
        yield server
    finally:
        # First close the transport to signal to the response_queue_to_buffer task that it should stop
        server._transport.close()
        for p in processes:
            p.terminate()
            p.join()
        manager.shutdown()


async def call_after_stream(streamer: AsyncIterator, callback, *args, **kwargs):
    try:
        async for item in streamer:
            yield item
    except Exception as e:
        logger.exception(f"Error in streamer: {e}")
    finally:
        callback(*args, **kwargs)


@dataclasses.dataclass
class WorkerSetupStatus:
    STARTING: str = "starting"
    READY: str = "ready"
    ERROR: str = "error"
    FINISHED: str = "finished"


def _get_default_handler(stream, format):
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    return handler


def configure_logging(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout, use_rich=False
):
    """Configure logging for the entire library with sensible defaults.

    Args:
        level (int): Logging level (default: logging.INFO)
        format (str): Log message format string
        stream (file-like): Output stream for logs
        use_rich (bool): Whether to use rich for logging

    """
    if use_rich:
        try:
            from rich.logging import RichHandler
            from rich.traceback import install

            install(show_locals=True)
            handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=True)
        except ImportError:
            logger.warning("Rich is not installed, using default logging")
            handler = _get_default_handler(stream, format)

    else:
        handler = _get_default_handler(stream, format)

    # Configure root library logger
    library_logger = logging.getLogger("litserve")
    library_logger.setLevel(level)
    library_logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    library_logger.propagate = False


def set_log_level(level):
    """Allow users to set the global logging level for the library."""
    logging.getLogger("litserve").setLevel(level)


def add_log_handler(handler):
    """Allow users to add custom log handlers.

    Example usage:
    file_handler = logging.FileHandler('library_logs.log')
    add_log_handler(file_handler)

    """
    logging.getLogger("litserve").addHandler(handler)


def generate_random_zmq_address(temp_dir="/tmp"):
    """Generate a random IPC address in the /tmp directory.

    Ensures the address is unique.
    Returns:
        str: A random IPC address suitable for ZeroMQ.

    """
    unique_name = f"zmq-{uuid.uuid4().hex}.ipc"
    ipc_path = os.path.join(temp_dir, unique_name)
    return f"ipc://{ipc_path}"


class ForkedPdb(pdb.Pdb):
    # Borrowed from - https://github.com/Lightning-AI/forked-pdb
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """

    def interaction(self, *args: Any, **kwargs: Any) -> None:
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")  # noqa: SIM115
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace():
    """Set a tracepoint in the code."""
    ForkedPdb().set_trace()


def set_trace_if_debug(debug_env_var="LITSERVE_DEBUG", debug_env_var_value="1"):
    """Set a tracepoint in the code if the environment variable LITSERVE_DEBUG is set."""
    if os.environ.get(debug_env_var) == debug_env_var_value:
        set_trace()


async def _stream_gen_from_thread(gen_func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=1)  # TODO: check if executor needs to be shutdown

    # This function runs in a separate thread
    def thread_generator():
        try:
            for item in gen_func(*args, **kwargs):
                # Block until the item is put in the queue
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
            # Signal completion
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
        except Exception as e:
            # Handle exceptions
            asyncio.run_coroutine_threadsafe(queue.put(e), loop).result()

    # Start the thread
    loop.run_in_executor(executor, thread_generator)

    # Yield items as they arrive
    while True:
        item = await queue.get()
        if item is None:
            break
        elif isinstance(item, Exception):
            raise item
        yield item


def asyncify(func):
    """Decorator that converts any function type to a consistent async interface.

    - Regular sync functions -> run in thread pool and return via coroutine
    - Sync generators -> converted to async generators that stream values
    - Async functions -> preserved as is
    - Async generators -> preserved as is

    Note: Uses functools.wraps to preserve the original function's signature and metadata.

    """
    # Already an async generator - return as is
    if inspect.isasyncgenfunction(func):
        return func

    # Already a coroutine function - return as is
    if asyncio.iscoroutinefunction(func):
        return func

    # Handle regular generator
    if inspect.isgeneratorfunction(func):

        @wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            return await _stream_gen_from_thread(func, *args, **kwargs)

        return async_gen_wrapper

    # Handle regular function
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    return async_wrapper
