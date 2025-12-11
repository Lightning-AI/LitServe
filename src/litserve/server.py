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
import contextlib
import copy
import inspect
import json
import logging
import multiprocessing as mp
import os
import pickle
import secrets
import socket
import sys
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from queue import Queue
from typing import TYPE_CHECKING, Literal, Optional, Union

import uvicorn
import uvicorn.server
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from starlette.formparsers import MultiPartParser
from starlette.middleware.gzip import GZipMiddleware

from litserve import LitAPI
from litserve.callbacks.base import Callback, CallbackRunner, EventTypes
from litserve.connector import _Connector
from litserve.loggers import Logger, _LoggerConnector
from litserve.loops import LitLoop, inference_worker
from litserve.middlewares import MaxSizeMiddleware, RequestCountMiddleware
from litserve.python_client import client_template
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.transport.factory import TransportConfig, create_transport_from_config
from litserve.utils import (
    LitAPIStatus,
    LoopResponseType,
    ResponseBufferItem,
    WorkerSetupStatus,
    add_ssl_context_from_env,
    call_after_stream,
    configure_logging,
    is_package_installed,
)

_MCP_AVAILABLE = is_package_installed("mcp")

if TYPE_CHECKING:
    from litserve.mcp import ToolEndpointType

mp.allow_connection_pickling()

logger = logging.getLogger(__name__)

# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")
SHUTDOWN_API_KEY = os.environ.get("LIT_SHUTDOWN_API_KEY")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# FastAPI writes form files to disk over 1MB by default, which prevents serialization by multiprocessing
MultiPartParser.max_file_size = sys.maxsize
# renamed in PR: https://github.com/encode/starlette/pull/2780
MultiPartParser.spool_max_size = sys.maxsize


def no_auth():
    pass


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


async def _mixed_response_to_buffer(
    transport: MessageTransport,
    response_buffer: dict[str, ResponseBufferItem],
    consumer_id: int = 0,
):
    """Handle both regular and streaming responses.

    Detect streaming responses by checking if the response is for streaming.

    """
    while True:
        try:
            result = await transport.areceive(consumer_id)
            if result is None:
                continue

            uid, (*response, response_type, worker_id) = result

            if response[1] == LitAPIStatus.START:
                response_buffer[uid].worker_id = int(worker_id)
                continue

            if response_type == LoopResponseType.STREAMING:
                response_buffer[uid].response_queue.append(response)
                response_buffer[uid].event.set()
            else:
                response_buffer[uid].response = response
                response_buffer[uid].event.set()
        except asyncio.CancelledError:
            logger.debug("Response queue to buffer task was cancelled")
            break
        except Exception as e:
            logger.error(f"Error in response_queue_to_buffer: {e}")
            break


async def response_queue_to_buffer(
    transport: MessageTransport,
    response_buffer: dict[str, ResponseBufferItem],
    consumer_id: int,
    litapi_connector: "_LitAPIConnector",
):
    mixed_streaming = (
        len(litapi_connector.lit_apis) > 1
        and litapi_connector.any_stream()
        and not all(api.stream for api in litapi_connector)
    )
    if mixed_streaming:
        return await _mixed_response_to_buffer(transport, response_buffer, consumer_id)

    stream = litapi_connector.any_stream()
    if stream:
        while True:
            try:
                result = await transport.areceive(consumer_id)
                if result is None:
                    continue

                uid, (*response, response_type, worker_id) = result

                if response[1] == LitAPIStatus.START:
                    response_buffer[uid].worker_id = int(worker_id)
                    continue

                response_buffer[uid].response_queue.append(response)
                response_buffer[uid].event.set()
            except asyncio.CancelledError:
                logger.debug("Response queue to buffer task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error in response_queue_to_buffer: {e}")
                break

    else:
        while True:
            try:
                result = await transport.areceive(consumer_id)
                if result is None:
                    continue

                uid, (*response, response_type, worker_id) = result

                if response[1] == LitAPIStatus.START:
                    response_buffer[uid].worker_id = int(worker_id)
                    continue

                response_buffer[uid].response = response
                response_buffer[uid].event.set()
            except asyncio.CancelledError:
                logger.debug("Response queue to buffer task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error in response_queue_to_buffer: {e}")
                break


def _migration_warning(feature_name):
    warnings.warn(
        f"The {feature_name} parameter is being deprecated in `LitServer` "
        "and will be removed in version v0.3.0.\n\n"
        "Please update your code to pass these arguments to `LitAPI` instead.\n\n"
        "Old usage:\n"
        f"    server = LitServer(api, {feature_name}=...)\n\n"
        "New usage:\n"
        f"    api = LitAPI({feature_name}=...)\n"
        "    server = LitServer(api, ...)",
        DeprecationWarning,
        stacklevel=3,
    )


class _LitAPIConnector:
    """A helper class to manage one or more `LitAPI` instances.

    This class provides utilities for performing setup tasks, managing request
    and batch timeouts, and interacting with `LitAPI` instances in a unified way.
    It ensures that all `LitAPI` instances are properly initialized and configured
    before use.

    Attributes:
        lit_apis (list[LitAPI]): A list of `LitAPI` instances managed by this connector.

    Methods:
        pre_setup(): Calls the `pre_setup` method on all managed `LitAPI` instances.
        set_request_timeout(timeout): Sets the request timeout for all `LitAPI` instances
            and validates that batch timeouts are within acceptable limits.
        __iter__(): Allows iteration over the managed `LitAPI` instances.
        any_stream(): Checks if any of the `LitAPI` instances have streaming enabled.
        set_logger_queue(queue): Sets a logger queue for all `LitAPI` instances.

    """

    def __init__(self, lit_apis: Union[LitAPI, Iterable[LitAPI]]):
        if isinstance(lit_apis, LitAPI):
            self.lit_apis = [lit_apis]
        elif isinstance(lit_apis, Iterable):
            self.lit_apis = list(lit_apis)
            if not self.lit_apis:  # Check if the iterable is empty
                raise ValueError("lit_apis must not be an empty iterable")
            self._detect_path_collision()
        else:
            raise ValueError(f"lit_apis must be a LitAPI or an iterable of LitAPI, but got {type(lit_apis)}")

    def _detect_path_collision(self):
        paths = {"/health": "LitServe healthcheck", "/info": "LitServe info"}
        for lit_api in self.lit_apis:
            if lit_api.api_path in paths:
                raise ValueError(f"api_path {lit_api.api_path} is already in use by {paths[lit_api.api_path]}")
            paths[lit_api.api_path] = lit_api

    def pre_setup(self):
        for lit_api in self.lit_apis:
            lit_api.pre_setup()
            # Ideally LitAPI should not know about LitLoop
            # LitLoop can keep litapi as a class variable
            lit_api.loop.pre_setup(lit_api)

    def set_request_timeout(self, timeout: float):
        for lit_api in self.lit_apis:
            lit_api.request_timeout = timeout

        for lit_api in self.lit_apis:
            if lit_api.batch_timeout > timeout and timeout not in (False, -1):
                raise ValueError("batch_timeout must be less than request_timeout")

    def __iter__(self):
        return iter(self.lit_apis)

    def any_stream(self):
        return any(lit_api.stream for lit_api in self.lit_apis)

    def set_logger_queue(self, queue: Queue):
        for lit_api in self.lit_apis:
            lit_api.set_logger_queue(queue)

    def get_mcp_tools(self) -> list["ToolEndpointType"]:
        mcp_tools = []
        for lit_api in self.lit_apis:
            if lit_api.mcp:
                mcp_tools.append(lit_api.mcp.as_tool())
        return mcp_tools


class BaseRequestHandler(ABC):
    def __init__(self, lit_api: LitAPI, server: "LitServer"):
        self.lit_api = lit_api
        self.server = server

    async def _prepare_request(self, request, request_type) -> dict:
        """Common request preparation logic."""
        if request_type == Request:
            content_type = request.headers.get("Content-Type", "")
            if content_type == "application/x-www-form-urlencoded" or content_type.startswith("multipart/form-data"):
                return await request.form()
            return await request.json()
        return request

    async def _submit_request(self, payload: dict) -> tuple[str, asyncio.Event]:
        """Submit request to worker queue."""
        request_queue = self.server._get_request_queue(self.lit_api.api_path)
        response_queue_id = self.server.app.response_queue_id
        uid = str(uuid.uuid4())

        # Trigger callback
        self.server._callback_runner.trigger_event(
            EventTypes.ON_REQUEST.value,
            active_requests=self.server.active_requests,
            litserver=self.server,
        )

        request_queue.put((response_queue_id, uid, time.monotonic(), payload))
        logger.debug(f"Submitted request uid={uid}")
        return uid, response_queue_id

    @abstractmethod
    async def handle_request(self, request, request_type) -> Response:
        pass


class RegularRequestHandler(BaseRequestHandler):
    async def handle_request(self, request, request_type) -> Response:
        try:
            logger.debug(f"Handling request: {request}")
            # Prepare request
            payload = await self._prepare_request(request, request_type)

            # Submit to worker
            uid, _ = await self._submit_request(payload)

            # Wait for response
            event = asyncio.Event()
            self.server.response_buffer[uid] = ResponseBufferItem(event)

            await event.wait()

            # Process response
            response_buffer_item = self.server.response_buffer.pop(uid)
            response, status = response_buffer_item.response

            if status == LitAPIStatus.ERROR:
                self._handle_error_response(response)

            # Trigger callback
            self.server._callback_runner.trigger_event(EventTypes.ON_RESPONSE.value, litserver=self.server)

            return response

        except HTTPException as e:
            raise e from None

        except Exception as e:
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error") from e

    @staticmethod
    def _handle_error_response(response):
        """Raise HTTPException as is and rest as 500 after logging the error."""
        try:
            if isinstance(response, bytes):
                response = pickle.loads(response)
                raise HTTPException(status_code=response.status_code, detail=response.detail)
        except Exception as e:
            logger.debug(f"couldn't unpickle error response {e}")

        if isinstance(response, HTTPException):
            raise response

        if isinstance(response, Exception):
            logger.error(f"Error while handling request: {response}")

        raise HTTPException(status_code=500, detail="Internal server error")


class StreamingRequestHandler(BaseRequestHandler):
    async def handle_request(self, request, request_type) -> StreamingResponse:
        try:
            # Prepare request
            payload = await self._prepare_request(request, request_type)

            # Submit to worker
            uid, _ = await self._submit_request(payload)

            # Set up streaming response
            event = asyncio.Event()
            response_queue = deque()
            self.server.response_buffer[uid] = ResponseBufferItem(event=event, response_queue=response_queue)

            # Create streaming response
            response_generator = call_after_stream(
                self.server.data_streamer(response_queue, data_available=event),
                self.server._callback_runner.trigger_event,
                EventTypes.ON_RESPONSE.value,
                litserver=self.server,
            )

            return StreamingResponse(response_generator)

        except Exception as e:
            logger.exception(f"Error handling streaming request: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


class _Server(uvicorn.server.Server):
    def run(self, worker_id: int, sockets: Union[list[socket.socket], None] = None) -> None:
        os.environ["LITSERVE_WORKER_ID"] = str(worker_id)
        return super().run(sockets)


class LitServer:
    """Initialize a LitServer for high-performance AI model serving.

    LitServer transforms AI models into production-ready APIs with automatic scaling,
    batching, streaming, and multi-GPU support.

    Quick Start:
        ```python
        import litserve as ls

        # Define inference pipeline
        class MyAPI(ls.LitAPI):
            def setup(self, device):
                self.model = load_model()  # model loading logic

            def predict(self, x):
                return self.model(x)

        # Create and run server
        server = ls.LitServer(MyAPI())
        server.run(port=8000)
        ```

    Args:
        lit_api:
            The core component - one or more LitAPI instances defining model logic.

            - Single API: `MyAPI()` for serving one model
            - Multiple APIs: `[API1(), API2()]` for multi-model serving

            Each LitAPI must implement:
            - `setup(device)`: Initialize the model
            - `predict(x)`: Run inference
            - Optional: `decode_request()`, `encode_response()` for custom I/O

    Hardware Configuration:
        accelerator:
            Hardware type for inference. Defaults to "auto".

            - "auto": Automatically detects best available (CUDA > MPS > CPU)
            - "cpu": Force CPU usage
            - "cuda": Use NVIDIA GPUs
            - "mps": Use Apple Metal Performance Shaders

        devices:
            Number of devices to use. Defaults to "auto".

            - "auto": Use all available devices
            - int: Use specific number (e.g., 2 for 2 GPUs)

        workers_per_device:
            Worker processes per device for parallel inference. Defaults to 1.

            - Higher values = better throughput but more memory usage
            - Good starting point: 1-4 depending on model size
            - For CPU, set to the number of cores available on the machine (e.g., 8 for 8-core CPU)
            - Monitor GPU memory when increasing

    Performance & Scaling:
        timeout:
            Request timeout in seconds. Defaults to 30.

            - Set to False or -1 to disable timeouts
            - Increase for slow models (e.g., 300 for large LLMs)
            - Decrease for fast models (e.g., 5 for lightweight models)

        fast_queue:
            Enable ZeroMQ for high-throughput scenarios (>100 RPS). Defaults to False.

            - Use when serving hundreds of requests per second
            - Not supported on Windows

        track_requests:
            Track active requests across all API servers for monitoring and load management. Defaults to False.

            When enabled, tracks the total number of active requests in the queue across all API servers
            and makes this count available via callbacks using the `on_request` hook. Essential for
            monitoring concurrent request load and implementing custom load management logic.

            - Recommended for production deployments
            - Access count via callbacks or `server.active_requests` property
            - Useful for monitoring and handling concurrent requests effectively

    API Configuration:
        healthcheck_path:
            Health check endpoint for load balancers. Defaults to "/health".

            - Returns 200 when all workers are ready
            - Critical for Kubernetes/Docker deployments

        info_path:
            Server information endpoint. Defaults to "/info".

            - Shows model metadata, device info, server config
            - Useful for debugging and monitoring

        disable_openapi_url:
            If True, disables the OpenAPI schema endpoint ("/openapi.json").

            - Useful for production environments where exposing API schemas is not desired.
            - Defaults to False (the OpenAPI schema is enabled).

        shutdown_path:
            Graceful shutdown endpoint. Defaults to "/shutdown".

        enable_shutdown_api:
            Enable remote shutdown capability. Defaults to False.

            - Requires authentication token (set LIT_SHUTDOWN_API_KEY env var)
            - Useful for automated deployment pipelines

        restart_workers:
            Enable this option to automatically restart
            workers if a critical error occurs. Defaults to False.

            - When enabled, the worker loop will exit using `os._exit(1)`,
            allowing the main process to recreate the worker.


    Content & Middleware:
        max_payload_size:
            Maximum request size. Defaults to "100MB".

            - String format: "10MB", "1GB"
            - Integer format: bytes (1048576 for 1MB)
            - Increase for large images/videos

        middlewares:
            HTTP middleware for cross-cutting concerns. Defaults to None.

            Example:
            ```python
            from starlette.middleware.cors import CORSMiddleware

            server = LitServer(
                api,
                middlewares=[
                    (CORSMiddleware, {"allow_origins": ["*"]}),
                    # Add more middleware as needed
                ]
            )
            ```

        model_metadata:
            Metadata about the model displayed at info endpoint. Defaults to None.

            Example:
            ```python
            metadata = {
                "model_name": "bert-base-uncased",
                "version": "1.0.0",
                "description": "Text classification model"
            }
            ```

    Monitoring & Debugging:
        callbacks:
            Event handlers for server lifecycle. Defaults to None.

            - Built-in callbacks for logging, metrics, custom logic
            - Triggers on request start/end, server start/stop

        loggers:
            Custom loggers for metrics and events. Defaults to None.

            - Integrate with monitoring stack
            - Track performance metrics, error rates

    Advanced Configuration:
        max_batch_size, batch_timeout, spec, stream, api_path, loop:
            **Deprecated**: Configure these in LitAPI implementation instead.

            Migration example:
            ```python
            # Old way (deprecated)
            server = LitServer(api, max_batch_size=8, stream=True)

            # New way (recommended)
            api = MyAPI(max_batch_size=8, stream=True)
            server = LitServer(api)
            ```

    Examples:
        Basic Usage:
        ```python
        import litserve as ls

        class SimpleAPI(ls.LitAPI):
            def setup(self, device):
                self.model = lambda x: x * 2  # model here

            def predict(self, x):
                return self.model(x)

        server = ls.LitServer(SimpleAPI())
        server.run()
        ```

        Production Setup:
        ```python
        server = ls.LitServer(
            MyAPI(max_batch_size=8),
            accelerator="cuda",
            devices=2,
            workers_per_device=4,
            fast_queue=True,
            track_requests=True,
            max_payload_size="50MB",
            timeout=60
        )
        server.run(port=8000, num_api_servers=4)
        ```

        Multi-Model Serving:
        ```python
        # Serve multiple models on different endpoints
        text_api = TextClassifierAPI(api_path="/classify")
        image_api = ImageClassifierAPI(api_path="/vision")

        server = ls.LitServer([text_api, image_api])
        server.run()
        ```

        Streaming Response:
        ```python
        class StreamingAPI(ls.LitAPI):
            def setup(self, device):
                self.model = load_llm()

            def predict(self, prompt):
                for token in self.model.generate(prompt):
                    yield {"token": token}

        server = ls.LitServer(StreamingAPI(stream=True))
        ```

    Deployment:
        Self-hosted:
        ```bash
        python server.py  # Run locally
        ```

        Lightning AI Cloud:
        ```bash
        lightning deploy server.py --cloud  # One-click deploy
        ```

    See Also:
        - LitAPI: Base class for defining model logic
        - LitSpec: API specifications (OpenAI compatibility)
        - Documentation: https://lightning.ai/docs/litserve

    """

    def __init__(
        self,
        lit_api: Union[LitAPI, list[LitAPI]],
        accelerator: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        devices: Union[int, Literal["auto"]] = "auto",
        workers_per_device: int = 1,
        timeout: Union[float, bool] = 30,
        healthcheck_path: str = "/health",
        info_path: str = "/info",
        shutdown_path: str = "/shutdown",
        enable_shutdown_api: bool = False,
        model_metadata: Optional[dict] = None,
        spec: Optional[LitSpec] = None,
        max_payload_size=None,
        track_requests: bool = False,
        callbacks: Optional[Union[list[Callback], Callback]] = None,
        middlewares: Optional[list[Union[Callable, tuple[Callable, dict]]]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None,
        fast_queue: bool = False,
        disable_openapi_url: bool = False,
        # All the following arguments are deprecated and will be removed in v0.3.0
        max_batch_size: Optional[int] = None,
        batch_timeout: float = 0.0,
        stream: bool = False,
        api_path: Optional[str] = None,
        loop: Optional[Union[str, LitLoop]] = None,
        restart_workers: bool = False,
    ):
        if max_batch_size is not None:
            warnings.warn(
                "'max_batch_size' and 'batch_timeout' are being deprecated in `LitServer` "
                "and will be removed in version v0.3.0.\n\n"
                "Please update your code to pass these arguments to `LitAPI` instead.\n\n"
                "Old usage:\n"
                "    server = LitServer(api, max_batch_size=N, batch_timeout=T, ...)\n\n"
                "New usage:\n"
                "    api = LitAPI(max_batch_size=N, batch_timeout=T, ...)\n"
                "    server = LitServer(api, ...)",
                DeprecationWarning,
                stacklevel=2,
            )
            lit_api.max_batch_size = max_batch_size
            lit_api.batch_timeout = batch_timeout

        if middlewares is None:
            middlewares = []
        if not isinstance(middlewares, list):
            _msg = (
                "middlewares must be a list of tuples"
                " where each tuple contains a middleware and its arguments. For example:\n"
                "server = ls.LitServer(ls.test_examples.SimpleLitAPI(), "
                'middlewares=[(RequestIdMiddleware, {"length": 5})])'
            )
            raise ValueError(_msg)

        # Handle 0.3.0 migration
        if api_path is not None:
            _migration_warning("api_path")
            lit_api.api_path = api_path
        if stream is True:
            _migration_warning("stream")
            lit_api.stream = stream
        if isinstance(loop, LitLoop):
            _migration_warning("loop")
            lit_api.loop = loop
        if isinstance(spec, LitSpec):
            _migration_warning("spec")
            lit_api.spec = spec
            lit_api.stream = spec.stream

        # pre setup
        self.litapi_connector = _LitAPIConnector(lit_api)
        self.litapi_connector.pre_setup()

        if api_path and not api_path.startswith("/"):
            raise ValueError(
                "api_path must start with '/'. "
                "Please provide a valid api path like '/predict', '/classify', or '/v1/predict'"
            )

        if not healthcheck_path.startswith("/"):
            raise ValueError(
                "healthcheck_path must start with '/'. "
                "Please provide a valid api path like '/health', '/healthcheck', or '/v1/health'"
            )

        if not info_path.startswith("/"):
            raise ValueError(
                "info_path must start with '/'. Please provide a valid api path like '/info', '/details', or '/v1/info'"
            )

        if enable_shutdown_api and not shutdown_path.startswith("/"):
            raise ValueError("shutdown_path must start with '/'. Please provide a valid api path like '/shutdown'")

        global SHUTDOWN_API_KEY
        if enable_shutdown_api and not SHUTDOWN_API_KEY:
            SHUTDOWN_API_KEY = secrets.token_urlsafe(32)
            logger.warning(
                "LitServe's Shutdown API is enabled, but the `LIT_SHUTDOWN_API_KEY` environment variable is missing."
                f"Generated shutdown API key: {SHUTDOWN_API_KEY}"
            )
        if enable_shutdown_api:
            curl_command = (
                "curl -X 'POST' 'http://localhost:8000/shutdown' "
                "-H 'accept: application/json' "
                f"-H 'Authorization: Bearer {SHUTDOWN_API_KEY}' "
                "-d ''"
            )
            logger.info(f"To shutdown the server, run command: \n{curl_command}\n")
        try:
            json.dumps(model_metadata)
        except (TypeError, ValueError):
            raise ValueError("model_metadata must be JSON serializable.")

        if sys.platform == "win32" and fast_queue:
            warnings.warn("ZMQ is not supported on Windows with LitServe. Disabling ZMQ.")
            fast_queue = False

        self.healthcheck_path = healthcheck_path
        self.info_path = info_path
        self._shutdown_path = shutdown_path
        self.track_requests = track_requests
        self.timeout = timeout
        self.litapi_connector.set_request_timeout(timeout)
        self.app = FastAPI(lifespan=self.lifespan, openapi_url="" if disable_openapi_url else "/openapi.json")
        self._disable_openapi_url = disable_openapi_url

        self.app.response_queue_id = None
        self.response_buffer: dict[str, ResponseBufferItem] = {}
        # gzip does not play nicely with streaming, see https://github.com/tiangolo/fastapi/discussions/8448
        if not self.litapi_connector.any_stream():
            middlewares.append((GZipMiddleware, {"minimum_size": 1000}))
        if max_payload_size is not None:
            middlewares.append((MaxSizeMiddleware, {"max_size": max_payload_size}))
        self.active_counters: list[mp.Value] = []
        self.middlewares = middlewares
        self._logger_connector = _LoggerConnector(self, loggers)
        self.logger_queue = None
        self.lit_api = lit_api
        self.enable_shutdown_api = enable_shutdown_api
        self.workers_per_device = workers_per_device
        self.max_payload_size = max_payload_size
        self.model_metadata = model_metadata
        self._connector = _Connector(accelerator=accelerator, devices=devices)
        self._callback_runner = CallbackRunner(callbacks)
        self.use_zmq = fast_queue
        self.transport_config = None
        self.litapi_request_queues = {}
        self._shutdown_event: Optional[mp.Event] = None
        self.uvicorn_graceful_timeout = 30
        self.restart_workers = restart_workers or False
        self.monitor_internal = 2
        self.mcp_server = None

        self._monitor_workers = True

        accelerator = self._connector.accelerator
        devices = self._connector.devices
        if accelerator == "cpu":
            self.devices = [accelerator]
        elif accelerator in ["cuda", "mps"]:
            device_list = devices
            if isinstance(devices, int):
                device_list = range(devices)
            self.devices = [self.device_identifiers(accelerator, device) for device in device_list]

        self.inference_workers_config = self.devices * self.workers_per_device
        self.transport_config = TransportConfig(transport_config="zmq" if self.use_zmq else "mp")
        self.register_endpoints()
        # register middleware
        self._register_middleware()

    def launch_inference_worker(self, lit_api: LitAPI):
        specs = [lit_api.spec] if lit_api.spec else []
        for spec in specs:
            # Objects of Server class are referenced (not copied)
            logging.debug(f"shallow copy for Server is created for for spec {spec}")
            server_copy = copy.copy(self)
            del server_copy.app, server_copy.transport_config, server_copy.litapi_connector
            spec.setup(server_copy)

        process_list = []
        endpoint = lit_api.api_path.split("/")[-1]
        for worker_id, device in enumerate(self.inference_workers_config):
            if len(device) == 1:
                device = device[0]

            self.workers_setup_status[f"{endpoint}_{worker_id}"] = WorkerSetupStatus.STARTING

            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=inference_worker,
                args=(
                    lit_api,
                    device,
                    worker_id,
                    self._get_request_queue(lit_api.api_path),
                    self._transport,
                    self.workers_setup_status,
                    self._callback_runner,
                    self.restart_workers,
                ),
                name="inference-worker",
            )
            process.start()
            process_list.append(process)
        return process_list

    def launch_single_inference_worker(self, lit_api: LitAPI, worker_id: int):
        specs = [lit_api.spec] if lit_api.spec else []
        for spec in specs:
            # Objects of Server class are referenced (not copied)
            logging.debug(f"shallow copy for Server is created for for spec {spec}")
            server_copy = copy.copy(self)
            del server_copy.app, server_copy.transport_config, server_copy.litapi_connector
            spec.setup(server_copy)

        device = self.inference_workers_config[worker_id]
        endpoint = lit_api.api_path.split("/")[-1]
        if len(device) == 1:
            device = device[0]

        self.workers_setup_status[f"{endpoint}_{worker_id}"] = WorkerSetupStatus.STARTING

        ctx = mp.get_context("spawn")
        process = ctx.Process(
            target=inference_worker,
            args=(
                lit_api,
                device,
                worker_id,
                self._get_request_queue(lit_api.api_path),
                self._transport,
                self.workers_setup_status,
                self._callback_runner,
                self.restart_workers,
            ),
            name="inference-worker",
        )

        process.start()
        return process

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        loop = asyncio.get_running_loop()

        if not hasattr(self, "_transport") or not self._transport:
            raise RuntimeError(
                "Response queues have not been initialized. "
                "Please make sure to call the 'launch_inference_workers' method of "
                "the LitServer class to initialize the response queues."
            )

        transport = self._transport
        future = response_queue_to_buffer(
            transport,
            self.response_buffer,
            app.response_queue_id,
            self.litapi_connector,
        )
        task = loop.create_task(future, name=f"response_queue_to_buffer-{app.response_queue_id}")
        task.add_done_callback(
            lambda _: logger.debug(f"Response queue to buffer task terminated for consumer_id {app.response_queue_id}")
        )

        try:
            if _MCP_AVAILABLE:
                async with self.mcp_server.lifespan(app):
                    yield
            else:
                yield
        finally:
            self._callback_runner.trigger_event(EventTypes.ON_SERVER_END.value, litserver=self)

            # Cancel the task
            task.cancel()

            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
                await asyncio.wait_for(task, timeout=1.0)

    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    @staticmethod
    async def data_streamer(q: deque, data_available: asyncio.Event, send_status: bool = False):
        while True:
            await data_available.wait()
            while len(q) > 0:
                data, status = q.popleft()
                if status == LitAPIStatus.FINISH_STREAMING:
                    return

                if status == LitAPIStatus.ERROR:
                    logger.error(
                        "Error occurred while streaming outputs from the inference worker. "
                        "Please check the above traceback."
                    )
                    if send_status:
                        yield data, status
                    return
                if send_status:
                    yield data, status
                else:
                    yield data
            data_available.clear()

    @property
    def active_requests(self):
        if self.track_requests and self.active_counters:
            return sum(counter.value for counter in self.active_counters)
        return None

    def _register_internal_endpoints(self):
        workers_ready = False

        @self.app.get("/", dependencies=[Depends(self.setup_auth())])
        async def index(request: Request) -> Response:
            return Response(content="litserve running")

        @self.app.get(self.healthcheck_path, dependencies=[Depends(self.setup_auth())])
        async def health(request: Request) -> Response:
            nonlocal workers_ready
            if not workers_ready:
                workers_ready = all(v == WorkerSetupStatus.READY for v in self.workers_setup_status.values())

            lit_api_health_status = True
            for lit_api in self.litapi_connector:
                result = lit_api.health()
                if inspect.isawaitable(result):
                    result = await result
                if not result:
                    lit_api_health_status = False
                    break
            if workers_ready and lit_api_health_status:
                return Response(content="ok", status_code=200)

            return Response(content="not ready", status_code=503)

        @self.app.get(self.info_path, dependencies=[Depends(self.setup_auth())])
        async def info(request: Request) -> Response:
            return JSONResponse(
                content={
                    "model": self.model_metadata,
                    "server": {
                        "devices": self.devices,
                        "workers_per_device": self.workers_per_device,
                        "timeout": self.timeout,
                        "stream": {lit_api.api_path: lit_api.stream for lit_api in self.litapi_connector},
                        "max_payload_size": self.max_payload_size,
                        "track_requests": self.track_requests,
                    },
                }
            )

        if self.enable_shutdown_api:

            @self.app.post(self._shutdown_path, dependencies=[Depends(self.shutdown_api_key_auth)])
            async def shutdown_endpoint():
                if not self._shutdown_event:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server is still starting up"
                    )
                self._shutdown_event.set()
                return Response(content="Server is initiating graceful shutdown.", status_code=status.HTTP_200_OK)

    def register_endpoints(self):
        self._register_internal_endpoints()

        for lit_api in self.litapi_connector:
            decode_request_signature = inspect.signature(lit_api.decode_request)
            encode_response_signature = inspect.signature(lit_api.encode_response)

            request_type = decode_request_signature.parameters["request"].annotation
            if request_type == decode_request_signature.empty:
                request_type = Request

            response_type = encode_response_signature.return_annotation
            if response_type == encode_response_signature.empty:
                response_type = Response
            self._register_api_endpoints(lit_api, request_type, response_type)

    def _get_request_queue(self, api_path: str):
        return self.litapi_request_queues[api_path]

    def _register_api_endpoints(self, lit_api: LitAPI, request_type, response_type):
        """Register endpoint routes for the FastAPI app."""

        self._callback_runner.trigger_event(EventTypes.ON_SERVER_START.value, litserver=self)

        # Create handlers
        handler = StreamingRequestHandler(lit_api, self) if lit_api.stream else RegularRequestHandler(lit_api, self)

        # Create endpoint function
        async def endpoint_handler(request: request_type) -> response_type:
            return await handler.handle_request(request, request_type)

        # Register endpoint
        if not lit_api.spec:
            self.app.add_api_route(
                lit_api.api_path,
                endpoint_handler,
                methods=["POST"],
                dependencies=[Depends(self.setup_auth())],
            )

        # Handle specs
        self._register_spec_endpoints(lit_api)

    def _register_spec_endpoints(self, lit_api: LitAPI):
        specs = [lit_api.spec] if lit_api.spec else []
        for spec in specs:
            spec: LitSpec
            # TODO check that path is not clashing
            for path, endpoint, methods in spec.endpoints:
                self.app.add_api_route(
                    path, endpoint=endpoint, methods=methods, dependencies=[Depends(self.setup_auth())]
                )

    def _register_middleware(self):
        for middleware in self.middlewares:
            if isinstance(middleware, tuple):
                middleware, kwargs = middleware
                self.app.add_middleware(middleware, **kwargs)
            elif callable(middleware):
                self.app.add_middleware(middleware)

    @staticmethod
    def generate_client_file(port: Union[str, int] = 8000):
        dest_path = os.path.join(os.getcwd(), "client.py")

        if os.path.exists(dest_path):
            logger.debug("client.py already exists in the current directory. Skipping generation.")
            return

        try:
            client_code = client_template.format(PORT=port)
            with open(dest_path, "w") as f:
                f.write(client_code)

        except Exception as e:
            logger.exception(f"Error copying file: {e}")

    def verify_worker_status(self):
        while not any(v == WorkerSetupStatus.READY for v in self.workers_setup_status.values()):
            if any(v == WorkerSetupStatus.ERROR for v in self.workers_setup_status.values()):
                raise RuntimeError("One or more workers failed to start. Shutting down LitServe")
            time.sleep(0.05)
        logger.debug("One or more workers are ready to serve requests")

    def _init_manager(self, num_api_servers: int):
        manager = mp.Manager()
        self.transport_config.manager = manager
        self.transport_config.num_consumers = num_api_servers
        self.workers_setup_status = manager.dict()
        self._shutdown_event = manager.Event()

        # create request queues for each unique lit_api api_path
        for lit_api in self.litapi_connector:
            self.litapi_request_queues[lit_api.api_path] = manager.Queue()

        if self._logger_connector._loggers:
            self.logger_queue = manager.Queue()
        self._logger_connector.run(self)
        self._transport = create_transport_from_config(self.transport_config)
        return manager

    def _perform_graceful_shutdown(
        self,
        manager: mp.Manager,
        uvicorn_workers: dict[str, Union[mp.Process, threading.Thread]],
        shutdown_reason: str = "normal",
    ):
        """Encapsulates the graceful shutdown logic."""
        logger.info("Shutting down LitServe...")

        # Handle transport closure based on shutdown reason
        if shutdown_reason == "keyboard_interrupt":
            logger.debug("KeyboardInterrupt detected - skipping transport cleanup to avoid hanging")
            self._transport.close(send_sentinel=False)
        else:
            self._transport.close(send_sentinel=True)

        # terminate Uvicorn server workers tracked by LitServe (the master processes/threads)
        if len(uvicorn_workers) > 0:
            for i, uw in uvicorn_workers.items():
                uvicorn_pid = uw.ident if isinstance(uw, threading.Thread) else uw.pid
                uvicorn_name = uw.name

                log_prefix = f"{uvicorn_name} (PID: {uvicorn_pid})"

                if not uw.is_alive():
                    logger.warning(f"{log_prefix}: Already not alive.")
                    continue
                try:
                    uw.terminate()
                    uw.join(timeout=self.uvicorn_graceful_timeout)
                    if uw.is_alive():
                        logger.warning(f"{log_prefix}: Did not terminate gracefully. Forcibly killing.")
                        uw.kill()
                except Exception as e:
                    logger.error(f"Error during termination of {log_prefix}: {e}")

        # terminate and join inference worker processes
        logger.debug(f"Terminating {len(self.inference_workers)} inference workers...")
        for i, iw in enumerate(self.inference_workers):
            worker_pid = iw.pid
            worker_name = iw.name
            try:
                iw.terminate()
                iw.join(timeout=5)
                if iw.is_alive():
                    logger.warning(
                        f"Worker {worker_name} (PID: {worker_pid}): Did not terminate gracefully. Killing (SIGKILL)."
                    )
                    iw.kill()
                else:
                    logger.debug(f"Worker {worker_name} (PID: {worker_pid}): Terminated gracefully.")
            except Exception as e:
                logger.error(f"Error while terminating worker {worker_name} (PID: {worker_pid}): {e}")

        manager.shutdown()

    def run(
        self,
        host: str = "0.0.0.0",
        port: Union[str, int] = 8000,
        num_api_servers: Optional[int] = None,
        log_level: str = "info",
        generate_client_file: bool = True,
        api_server_worker_type: Literal["process", "thread"] = "process",
        pretty_logs: bool = False,
        **kwargs,
    ):
        """Start the LitServer to serve AI model requests with production-ready performance.

        This method launches the complete serving infrastructure: initializes worker processes,
        starts the HTTP server, and begins handling requests. The server runs until manually
        stopped (Ctrl+C) or programmatically shut down.

        Quick Start:
            ```python
            # Basic usage - starts server on localhost:8000
            server.run()

            # Production - multiple servers and custom port
            server.run(port=8080, num_api_servers=4)
            ```

        Server Lifecycle:
            1. **Initialize**: Sets up worker processes and communication queues
            2. **Health Check**: Verifies all workers are ready to serve requests
            3. **Start HTTP Server**: Begins accepting requests on specified host/port
            4. **Serve Requests**: Distributes requests to workers and returns responses
            5. **Graceful Shutdown**: Properly terminates workers when stopped

        Args:
            host:
                Network interface to bind the server to. Defaults to "0.0.0.0".

                - "0.0.0.0": Accept connections from any IP (public access)
                - "127.0.0.1": Only accept local connections (localhost only)
                - "::": IPv6 equivalent of "0.0.0.0"

                For development, use "127.0.0.1" for security. For production/Docker, use "0.0.0.0".

            port:
                Port number to listen on. Defaults to 8000.

                - Must be between 1024-65535 (privileged ports require admin)
                - Ensure the port is available and not blocked by firewalls
                - Common choices: 8000, 8080, 3000, 5000

        Performance Configuration:
            num_api_servers:
                Number of parallel HTTP server processes. Defaults to None (auto-detect).

                - None: Uses same count as inference workers (recommended)
                - Higher values improve HTTP throughput but use more memory
                - Good starting point: 2-8 depending on expected load
                - Each server handles HTTP requests independently

            api_server_worker_type:
                Process architecture for HTTP servers. Defaults to "process".

                - "process": Better isolation, CPU utilization, and fault tolerance
                - "thread": Lower memory usage but shared memory space
                - Windows automatically uses "thread" (process forking not supported)

        Development & Debugging:
            log_level:
                Logging verbosity level. Defaults to "info".

                - "critical": Only severe errors
                - "error": Error conditions
                - "warning": Warning messages (good for production)
                - "info": General information (default)
                - "debug": Detailed debugging info (development)
                - "trace": Very verbose output (troubleshooting)

            pretty_logs:
                Enable enhanced log formatting with colors and rich formatting. Defaults to False.

                - Requires: `pip install rich`
                - Great for development and local debugging
                - May not display properly in some production log aggregators

            generate_client_file:
                Auto-generate a Python client file for easy API interaction. Defaults to True.

                - Creates `client.py` in current directory with typed methods
                - Useful for testing and integration
                - Safe to disable in production environments

        Advanced Configuration:
            **kwargs:
                Additional uvicorn server configuration options.

                Common SSL options:
                ```python
                server.run(
                    ssl_keyfile="path/to/key.pem",
                    ssl_certfile="path/to/cert.pem"
                )
                ```

                Other uvicorn options: ssl_ca_certs, ssl_ciphers, ssl_version,
                workers, backlog, etc. See uvicorn documentation for full list.

        Examples:
            Basic Development:
            ```python
            # Simple local development
            server.run()
            # Access at: http://localhost:8000
            # API docs at: http://localhost:8000/docs
            ```

            Production Configuration:
            ```python
            # High-performance production setup
            server.run(
                host="0.0.0.0",
                port=8000,
                num_api_servers=8,
                log_level="warning",
                pretty_logs=False,
                generate_client_file=False
            )
            ```

            Development with Debug:
            ```python
            # Development with detailed logging
            server.run(
                host="127.0.0.1",
                port=8000,
                log_level="debug",
                pretty_logs=True,
                num_api_servers=1
            )
            ```

            Multi-API Server:
            ```python
            # Balance load across multiple HTTP servers
            server.run(
                port=8000,
                num_api_servers=4,  # 4 parallel HTTP servers
                api_server_worker_type="process"
            )
            ```

        Server Endpoints:
            Once running, the server provides several built-in endpoints:

            - **Main API**: `POST /predict` (or custom path from LitAPI)
            - **Health Check**: `GET /health` - Returns 200 when ready
            - **Server Info**: `GET /info` - Shows configuration and metadata
            - **API Documentation**: `GET /docs` - Interactive Swagger UI
            - **OpenAPI Schema**: `GET /openapi.json` - API specification

        Stopping the Server:
            - **Ctrl+C**: Graceful shutdown (recommended)
            - **SIGTERM**: Graceful shutdown in Docker/Kubernetes
            - **Shutdown API**: POST to `/shutdown` (if enabled)

        Common Issues:
            - **Port in use**: Choose different port or stop conflicting process
            - **Permission denied**: Use port > 1024 or run with appropriate permissions
            - **Workers not ready**: Check model loading in LitAPI.setup() method
            - **Memory issues**: Reduce num_api_servers or workers_per_device

        Notes:
            - Server blocks execution until stopped (use threads for non-blocking)
            - Logs show startup progress and any configuration issues
            - Swagger UI provides interactive API testing interface

        """
        if generate_client_file:
            LitServer.generate_client_file(port=port)

        port_msg = f"port must be a value from 1024 to 65535 but got {port}"
        try:
            port = int(port)
        except ValueError:
            raise ValueError(port_msg)

        if not (1024 <= port <= 65535):
            raise ValueError(port_msg)

        host_msg = f"host must be '0.0.0.0', '127.0.0.1', or '::' but got {host}"
        if host not in ["0.0.0.0", "127.0.0.1", "::"]:
            raise ValueError(host_msg)

        kwargs = add_ssl_context_from_env(kwargs)

        configure_logging(log_level, use_rich=pretty_logs)
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level=log_level,
            **kwargs,
        )
        sockets = [config.bind_socket()]

        if num_api_servers is None:
            num_api_servers = len(self.inference_workers_config)

        if num_api_servers < 1:
            raise ValueError("num_api_servers must be greater than 0")

        if sys.platform == "win32":
            warnings.warn(
                "Windows does not support forking. Using threads api_server_worker_type will be set to 'thread'"
            )
            api_server_worker_type = "thread"
        elif api_server_worker_type is None:
            api_server_worker_type = "process"

        manager = self._init_manager(num_api_servers)
        self._logger_connector.run(self)
        self.inference_workers = []
        for lit_api in self.litapi_connector:
            _inference_workers = self.launch_inference_worker(lit_api)
            self.inference_workers.extend(_inference_workers)

        self.verify_worker_status()

        shutdown_reason = "normal"
        uvicorn_workers = {}
        try:
            uvicorn_workers = self._start_server(
                port, num_api_servers, log_level, sockets, api_server_worker_type, **kwargs
            )

            if not self._disable_openapi_url:
                print(f"Swagger UI is available at http://0.0.0.0:{port}/docs")

            if self._monitor_workers:
                self._start_worker_monitoring(manager, uvicorn_workers)

            self._shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Initiating graceful shutdown.")
            shutdown_reason = "keyboard_interrupt"
        finally:
            self._perform_graceful_shutdown(manager, uvicorn_workers, shutdown_reason)

    def _prepare_app_run(self, app: FastAPI):
        if self.track_requests:
            # create a copy of the middleware list
            app.user_middleware = list(app.user_middleware)

            # Add middleware to count active requests
            active_counter = mp.Value("i", 0, lock=True)
            self.active_counters.append(active_counter)
            app.add_middleware(RequestCountMiddleware, active_counter=active_counter)

    def _start_server(self, port, num_uvicorn_servers, log_level, sockets, uvicorn_worker_type, **kwargs):
        workers = []
        for response_queue_id in range(num_uvicorn_servers):
            self.app.response_queue_id = response_queue_id
            for lit_api in self.litapi_connector:
                if lit_api.spec:
                    lit_api.spec.response_queue_id = response_queue_id

            if _MCP_AVAILABLE:
                from litserve.mcp import _LitMCPServerConnector

                self.mcp_server = _LitMCPServerConnector()
                self.mcp_server.connect_mcp_server(self.litapi_connector.get_mcp_tools(), self.app)
            app: FastAPI = copy.copy(self.app)

            self._prepare_app_run(app)
            uvicorn_config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=port,
                log_level=log_level,
                timeout_graceful_shutdown=self.uvicorn_graceful_timeout,
                **kwargs,
            )
            if sys.platform == "win32" and num_uvicorn_servers > 1:
                logger.debug("Enable Windows explicit socket sharing...")
                # We make sure sockets is listening...
                # It prevents further [WinError 10022]
                for sock in sockets:
                    sock.listen(uvicorn_config.backlog)
                # We add worker to say unicorn to use a shared socket (win32)
                # https://github.com/encode/uvicorn/pull/802
                uvicorn_config.workers = num_uvicorn_servers

            server = _Server(config=uvicorn_config)
            if uvicorn_worker_type == "process":
                ctx = mp.get_context("fork")
                w = ctx.Process(
                    target=server.run, args=(response_queue_id, sockets), name=f"LitServer-{response_queue_id}"
                )
            elif uvicorn_worker_type == "thread":
                w = threading.Thread(
                    target=server.run, args=(response_queue_id, sockets), name=f"LitServer-{response_queue_id}"
                )
            else:
                raise ValueError("Invalid value for api_server_worker_type. Must be 'process' or 'thread'")
            w.start()
            workers.append(w)
        return dict(enumerate(workers))

    def setup_auth(self):
        if hasattr(self.lit_api, "authorize") and callable(self.lit_api.authorize):
            return self.lit_api.authorize
        if LIT_SERVER_API_KEY:
            return api_key_auth
        return no_auth

    def shutdown_api_key_auth(self, shutdown_api_key: str = Depends(oauth2_scheme)):
        if not SHUTDOWN_API_KEY or shutdown_api_key != SHUTDOWN_API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid Bearer token for Shutdown API."
                " Check that you are passing a correct 'Authorization: Bearer <SHUTDOWN_API_KEY>' in your header.",
            )

    def _start_worker_monitoring(
        self,
        manager: mp.Manager,
        uvicorn_workers: dict[str, Union[mp.Process, threading.Thread]],
    ):
        def monitor():
            try:
                while not self._shutdown_event.is_set():
                    broken_workers = {}

                    for i, proc in enumerate(self.inference_workers):
                        if proc.is_alive():
                            continue

                        if not self.restart_workers:
                            logger.error(f"⚠️ Worker {proc.name} died; shutting down")
                            self._perform_graceful_shutdown(manager, uvicorn_workers, f"⚠️ Worker {proc.name} died)")
                            return

                        broken_workers[i] = proc

                    for idx, proc in broken_workers.items():
                        lit_api_id = idx // len(self.inference_workers_config)
                        worker_id = idx % len(self.inference_workers_config)

                        for uid, resp in self.response_buffer.items():
                            if resp.worker_id is None or resp.worker_id != worker_id:
                                continue

                            # Check whether the event has already been set
                            if resp.event.is_set():
                                continue

                            resp.response = (None, LitAPIStatus.ERROR)

                            if resp.response_queue is not None:
                                resp.response_queue.append((None, LitAPIStatus.ERROR))

                            resp.event.set()
                            print(f"[monoriting] Marked {uid} set")

                        print(f"[monoriting] Worker {worker_id} is dead. Restarting it")
                        lit_api = self.litapi_connector.lit_apis[lit_api_id]
                        self.inference_workers[idx] = self.launch_single_inference_worker(lit_api, worker_id)
                        print(f"[monoriting] Worker {worker_id} has been started.")

                    time.sleep(self.monitor_internal)

            except Exception as e:
                print(e)

        t = threading.Thread(target=monitor, daemon=True, name="litserve-monitoring")
        t.start()
