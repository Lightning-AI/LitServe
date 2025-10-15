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
import base64
import dataclasses
import importlib.util
import logging
import os
import pdb
import pickle
import sys
import tempfile
import time
import uuid
import warnings
from abc import ABCMeta
from collections.abc import AsyncIterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, Union

from fastapi import HTTPException

if TYPE_CHECKING:
    from litserve.server import LitServer

logger = logging.getLogger(__name__)

_DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(processName)s[%(process)d] - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
# Threshold for detecting heavy initialization tasks.
# A value of 1 second was chosen based on empirical observations
# of typical initialization times in this project.
_INIT_THRESHOLD = 1


class LitAPIStatus:
    START = "START"
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
def wrap_litserve_start(server: "LitServer", worker_monitor: bool = False):
    """Pytest utility to start the server in a context manager."""
    server.app.response_queue_id = 0
    for lit_api in server.litapi_connector:
        if lit_api.spec:
            lit_api.spec.response_queue_id = 0

    server.manager = server._init_manager(1)
    
    server.inference_workers = []
    for lit_api in server.litapi_connector:
        server.inference_workers.extend(server.launch_inference_workers(lit_api))
    
    server._prepare_app_run(server.app)
    
    if worker_monitor:
        server._start_worker_monitoring(server.manager, {})
    
    if is_package_installed("mcp"):
        from litserve.mcp import _LitMCPServerConnector

        server.mcp_server = _LitMCPServerConnector()
    else:
        server.mcp_server = None

    try:
        yield server
    finally:
        server._shutdown_event.set()
        # First close the transport to signal to the response_queue_to_buffer task that it should stop
        server._transport.close()
        for p in server.inference_workers:
            p.terminate()
            p.join()
        server.manager.shutdown()


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
    format: str = _DEFAULT_LOG_FORMAT,
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


def is_package_installed(package_name: str) -> bool:
    spec = importlib.util.find_spec(package_name)
    return spec is not None


class _TimedInitMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls._has_custom_setup = False

        for base in bases:
            if hasattr(base, "setup"):
                base_setup = base.setup

                if "setup" in namespace and namespace["setup"] is not base_setup:
                    cls._has_custom_setup = True
                    break
        else:
            if "setup" in namespace:
                cls._has_custom_setup = True

        return cls

    def __call__(cls, *args, **kwargs):
        start_time = time.perf_counter()
        instance = super().__call__(*args, **kwargs)
        elapsed = time.perf_counter() - start_time

        if elapsed >= _INIT_THRESHOLD and not cls._has_custom_setup:
            warnings.warn(
                (
                    f"{cls.__name__}.__init__ took {elapsed:.2f} seconds to execute. This suggests that you're "
                    "loading a model or doing other heavy processing inside the constructor.\n\n"
                    "To improve startup performance and avoid unnecessary work across processes, move any one-time "
                    f"heavy initialization into the `{cls.__name__}.setup` method.\n\n"
                    "The `LitAPI.setup` method is designed for deferred, process-specific loading — ideal for models "
                    "and large resources."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        return instance


def add_ssl_context_from_env(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Loads SSL context from base64-encoded environment variables.

    This function checks for the presence of `LIGHTNING_CERT_PEM` and
    `LIGHTNING_KEY_FILE` environment variables. It expects these variables
    to contain the SSL certificate and private key, respectively, as
    base64-encoded PEM strings.

    If both variables are found, it decodes them and writes the content to
    secure, temporary files. The paths to these files are returned in a
    dictionary suitable for direct use as keyword arguments in libraries
    that require SSL file paths (like `uvicorn` or `requests`).

    Note:
        The temporary files are not automatically deleted (`delete=False`).
        The calling application is responsible for cleaning up these files
        after the SSL context is no longer needed to prevent leaving
        sensitive data on disk.

    Returns:
        dict[str, Any]: A dictionary containing `ssl_certfile` and `ssl_keyfile`
        keys with `pathlib.Path` objects pointing to the temporary files.
        If either of the required environment variables is missing, it
        returns an empty dictionary.

    """

    if "ssl_keyfile" in kwargs and "ssl_certfile" in kwargs:
        return kwargs

    cert_pem_b64 = os.getenv("LIGHTNING_CERT_PEM", "")
    cert_key_b64 = os.getenv("LIGHTNING_KEY_FILE", "")

    if cert_pem_b64 == "" or cert_key_b64 == "":
        return kwargs

    # Decode the base64 strings to get the actual PEM content
    cert_pem = base64.b64decode(cert_pem_b64).decode("utf-8")
    cert_key = base64.b64decode(cert_key_b64).decode("utf-8")

    # Write to temporary files
    with (
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as cert_file,
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as key_file,
    ):
        cert_file.write(cert_pem)
        cert_file.flush()
        key_file.write(cert_key)
        key_file.flush()

        logger.info("Loading TLS Certificates \n")

        # Return a dictionary with Path objects to the created files
        return {"ssl_keyfile": Path(key_file.name), "ssl_certfile": Path(cert_file.name), **kwargs}
