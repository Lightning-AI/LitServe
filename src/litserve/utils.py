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
import logging
import os
import pdb
import pickle
import sys
import time
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, TextIO, Union

from fastapi import HTTPException

if TYPE_CHECKING:
    from litserve.server import LitServer

logger = logging.getLogger(__name__)


class LitAPIStatus:
    OK = "OK"
    ERROR = "ERROR"
    FINISH_STREAMING = "FINISH_STREAMING"


class LoopResponseType(Enum):
    STREAMING = "STREAMING"
    REGULAR = "REGULAR"


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


@contextmanager
def test_litserve_shutdown(server: "LitServer"):
    """Pytest utility to start the server in a context manager and perform graceful shutdown."""
    # These lines are related to Uvicorn workers, which TestClient doesn't fully simulate.
    # They might be removed or adapted if they cause issues with TestClient lifecycle.
    server.app.response_queue_id = 0
    for lit_api in server.litapi_connector:
        if lit_api.spec:
            lit_api.spec.response_queue_id = 0

    # Initialize manager and launch workers as done in server.run()
    server._init_manager(num_api_servers=1)  # Assume 1 API server for TestClient context

    for lit_api in server.litapi_connector:
        server.launch_inference_worker(lit_api)

    # Verify workers are ready before yielding to the test
    server.verify_worker_status()

    # Prepare the app with middleware, as done in _start_server
    server._prepare_app_run(server.app)

    try:
        yield server
    finally:
        # Trigger the shutdown event (if the test didn't already)
        if server._shutdown_event and not server._shutdown_event.is_set():
            server._shutdown_event.set()
            logger.info("Shutdown event explicitly set by context manager teardown.")

        # Use the server's built-in graceful shutdown logic
        server._perform_graceful_shutdown()
        logger.info("LitServer gracefully shut down by context manager.")

        # Give a small moment for logs to flush before process exits
        time.sleep(0.5)


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
    level: Union[str, int] = logging.INFO,
    format: str = "%(asctime)s - %(processName)s[%(process)d] - %(name)s - %(levelname)s - %(message)s",
    stream: TextIO = sys.stdout,
    use_rich: bool = False,
):
    """Configure logging for the entire library with sensible defaults.

    Args:
        level (int): Logging level (default: logging.INFO)
        format (str): Log message format string
        stream (file-like): Output stream for logs
        use_rich (bool): Makes the logs more readable by using rich, useful for debugging. Defaults to False.

    """
    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level)

    # Clear any existing handlers to prevent duplicates
    library_logger = logging.getLogger("litserve")
    for handler in library_logger.handlers[:]:
        library_logger.removeHandler(handler)

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

    # Configure library logger
    library_logger.setLevel(level)
    library_logger.addHandler(handler)
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
