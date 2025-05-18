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
import pickle
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, AsyncIterator

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
    server.app.response_queue_id = 0
    if server.lit_spec:
        server.lit_spec.response_queue_id = 0
    manager, processes = server.launch_inference_worker(num_uvicorn_servers=1)
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


def configure_logging(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout
):
    """Configure logging for the entire library with sensible defaults.

    Args:
        level (int): Logging level (default: logging.INFO)
        format (str): Log message format string
        stream (file-like): Output stream for logs

    """
    # Create a library-wide handler
    handler = logging.StreamHandler(stream)

    # Set formatter with user-configurable format
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

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


async def _stream_gen_from_thread(gen_func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=1)

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

    Works with:
    - Regular sync functions (runs in thread pool)
    - Sync generators (streams through async generator)
    - Async functions (preserves behavior)
    - Async generators (preserves behavior)

    """

    async def wrapper(*args, **kwargs):
        if inspect.isgeneratorfunction(func):
            return _stream_gen_from_thread(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)

        if inspect.isasyncgenfunction(func):
            return func(*args, **kwargs)

        # Handle regular functions in thread
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    return wrapper
