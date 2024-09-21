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
import copy
import inspect
import logging
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
import uuid
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from queue import Empty
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.formparsers import MultiPartParser
from starlette.middleware.gzip import GZipMiddleware

from litserve import LitAPI
from litserve.callbacks.base import CallbackRunner, Callback, EventTypes
from litserve.connector import _Connector
from litserve.loops import inference_worker
from litserve.specs import OpenAISpec
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, MaxSizeMiddleware, load_and_raise

mp.allow_connection_pickling()

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

except ImportError:
    print("uvloop is not installed. Falling back to the default asyncio event loop.")

logger = logging.getLogger(__name__)

# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")

# FastAPI writes form files to disk over 1MB by default, which prevents serialization by multiprocessing
MultiPartParser.max_file_size = sys.maxsize


def no_auth():
    pass


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


async def response_queue_to_buffer(
    response_queue: mp.Queue,
    response_buffer: Dict[str, Union[Tuple[deque, asyncio.Event], asyncio.Event]],
    stream: bool,
    threadpool: ThreadPoolExecutor,
):
    loop = asyncio.get_running_loop()
    if stream:
        while True:
            try:
                uid, response = await loop.run_in_executor(threadpool, response_queue.get)
            except Empty:
                await asyncio.sleep(0.0001)
                continue
            stream_response_buffer, event = response_buffer[uid]
            stream_response_buffer.append((uid, response))
            event.set()

    else:
        while True:
            uid, response = await loop.run_in_executor(threadpool, response_queue.get)
            event = response_buffer.pop(uid)
            response_buffer[uid] = response
            event.set()


class LitServer:
    def __init__(
        self,
        lit_api: LitAPI,
        accelerator: str = "auto",
        devices: Union[str, int] = "auto",
        workers_per_device: int = 1,
        timeout: Union[float, bool] = 30,
        max_batch_size: int = 1,
        batch_timeout: float = 0.0,
        api_path: str = "/predict",
        stream: bool = False,
        spec: Optional[LitSpec] = None,
        max_payload_size=None,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        middlewares: Optional[list[Union[Callable, tuple[Callable, dict]]]] = None,
    ):
        if batch_timeout > timeout and timeout not in (False, -1):
            raise ValueError("batch_timeout must be less than timeout")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")
        if isinstance(spec, OpenAISpec):
            stream = True

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

        if not api_path.startswith("/"):
            raise ValueError(
                "api_path must start with '/'. "
                "Please provide a valid api path like '/predict', '/classify', or '/v1/predict'"
            )

        # Check if the batch and unbatch methods are overridden in the lit_api instance
        batch_overridden = lit_api.batch.__code__ is not LitAPI.batch.__code__
        unbatch_overridden = lit_api.unbatch.__code__ is not LitAPI.unbatch.__code__

        if batch_overridden and unbatch_overridden and max_batch_size == 1:
            warnings.warn(
                "The LitServer has both batch and unbatch methods implemented, "
                "but the max_batch_size parameter was not set."
            )

        self.api_path = api_path
        lit_api.stream = stream
        lit_api.request_timeout = timeout
        lit_api._sanitize(max_batch_size, spec=spec)
        self.app = FastAPI(lifespan=self.lifespan)
        self.app.response_queue_id = None
        self.response_queue_id = None
        self.response_buffer = {}
        # gzip does not play nicely with streaming, see https://github.com/tiangolo/fastapi/discussions/8448
        if not stream:
            middlewares.append((GZipMiddleware, {"minimum_size": 1000}))
        if max_payload_size is not None:
            middlewares.append((MaxSizeMiddleware, {"max_size": max_payload_size}))
        self.middlewares = middlewares
        self.lit_api = lit_api
        self.lit_spec = spec
        self.workers_per_device = workers_per_device
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.stream = stream
        self.max_payload_size = max_payload_size
        self._connector = _Connector(accelerator=accelerator, devices=devices)
        self._callback_runner = CallbackRunner(callbacks)

        specs = spec if spec is not None else []
        self._specs = specs if isinstance(specs, Sequence) else [specs]

        decode_request_signature = inspect.signature(lit_api.decode_request)
        encode_response_signature = inspect.signature(lit_api.encode_response)

        self.request_type = decode_request_signature.parameters["request"].annotation
        if self.request_type == decode_request_signature.empty:
            self.request_type = Request

        self.response_type = encode_response_signature.return_annotation
        if self.response_type == encode_response_signature.empty:
            self.response_type = Response

        accelerator = self._connector.accelerator
        devices = self._connector.devices
        if accelerator == "cpu":
            self.devices = [accelerator]
        elif accelerator in ["cuda", "mps"]:
            device_list = devices
            if isinstance(devices, int):
                device_list = range(devices)
            self.devices = [self.device_identifiers(accelerator, device) for device in device_list]

        self.inference_workers = self.devices * self.workers_per_device
        self.register_endpoints()

    def launch_inference_worker(self, num_uvicorn_servers: int):
        manager = mp.Manager()
        self.workers_setup_status = manager.dict()
        self.request_queue = manager.Queue()
        self.request_evicted_status = manager.dict()

        self.response_queues = [manager.Queue() for _ in range(num_uvicorn_servers)]

        for spec in self._specs:
            # Objects of Server class are referenced (not copied)
            logging.debug(f"shallow copy for Server is created for for spec {spec}")
            server_copy = copy.copy(self)
            del server_copy.app
            spec.setup(server_copy)

        process_list = []
        for worker_id, device in enumerate(self.inference_workers):
            if len(device) == 1:
                device = device[0]

            self.workers_setup_status[worker_id] = False

            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=inference_worker,
                args=(
                    self.lit_api,
                    self.lit_spec,
                    device,
                    worker_id,
                    self.request_queue,
                    self.response_queues,
                    self.max_batch_size,
                    self.batch_timeout,
                    self.stream,
                    self.workers_setup_status,
                    self.request_evicted_status,
                    self._callback_runner,
                ),
            )
            process.start()
            process_list.append(process)
        return manager, process_list

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        loop = asyncio.get_running_loop()

        if not hasattr(self, "response_queues") or not self.response_queues:
            raise RuntimeError(
                "Response queues have not been initialized. "
                "Please make sure to call the 'launch_inference_worker' method of "
                "the LitServer class to initialize the response queues."
            )

        response_queue = self.response_queues[app.response_queue_id]
        response_executor = ThreadPoolExecutor(max_workers=len(self.inference_workers))
        future = response_queue_to_buffer(response_queue, self.response_buffer, self.stream, response_executor)
        task = loop.create_task(future)

        yield

        self._callback_runner.trigger_event(EventTypes.ON_SERVER_END, litserver=self)
        task.cancel()
        logger.debug("Shutting down response queue to buffer task")

    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    async def data_streamer(self, q: deque, data_available: asyncio.Event, send_status: bool = False):
        uid = None
        while True:
            try:
                await data_available.wait()
                while len(q) > 0:
                    uid, (data, status) = q.popleft()
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
            except asyncio.CancelledError:
                if uid is not None:
                    self.request_evicted_status[uid] = True
                    logger.exception("Streaming request cancelled for the uid=%s", uid)
                return

    def register_endpoints(self):
        """Register endpoint routes for the FastAPI app and setup middlewares."""
        self._callback_runner.trigger_event(EventTypes.ON_SERVER_START, litserver=self)
        workers_ready = False

        @self.app.get("/", dependencies=[Depends(self.setup_auth())])
        async def index(request: Request) -> Response:
            return Response(content="litserve running")

        @self.app.get("/health", dependencies=[Depends(self.setup_auth())])
        async def health(request: Request) -> Response:
            nonlocal workers_ready
            if not workers_ready:
                workers_ready = all(self.workers_setup_status.values())

            if workers_ready:
                return Response(content="ok", status_code=200)

            return Response(content="not ready", status_code=503)

        async def predict(request: self.request_type) -> self.response_type:
            response_queue_id = self.app.response_queue_id
            uid = uuid.uuid4()
            event = asyncio.Event()
            self.response_buffer[uid] = event
            logger.info(f"Received request uid={uid}")

            payload = request
            if self.request_type == Request:
                if request.headers["Content-Type"] == "application/x-www-form-urlencoded" or request.headers[
                    "Content-Type"
                ].startswith("multipart/form-data"):
                    payload = await request.form()
                else:
                    payload = await request.json()

            self.request_queue.put_nowait((response_queue_id, uid, time.monotonic(), payload))

            await event.wait()
            response, status = self.response_buffer.pop(uid)

            if status == LitAPIStatus.ERROR:
                load_and_raise(response)
            return response

        async def stream_predict(request: self.request_type) -> self.response_type:
            response_queue_id = self.app.response_queue_id
            uid = uuid.uuid4()
            event = asyncio.Event()
            q = deque()
            self.response_buffer[uid] = (q, event)
            logger.debug(f"Received request uid={uid}")

            payload = request
            if self.request_type == Request:
                payload = await request.json()
            self.request_queue.put((response_queue_id, uid, time.monotonic(), payload))

            return StreamingResponse(self.data_streamer(q, data_available=event))

        if not self._specs:
            stream = self.lit_api.stream
            # In the future we might want to differentiate endpoints for streaming vs non-streaming
            # For now we allow either one or the other
            endpoint = self.api_path
            methods = ["POST"]
            self.app.add_api_route(
                endpoint,
                stream_predict if stream else predict,
                methods=methods,
                dependencies=[Depends(self.setup_auth())],
            )

        for spec in self._specs:
            spec: LitSpec
            # TODO check that path is not clashing
            for path, endpoint, methods in spec.endpoints:
                self.app.add_api_route(
                    path, endpoint=endpoint, methods=methods, dependencies=[Depends(self.setup_auth())]
                )

        for middleware in self.middlewares:
            if isinstance(middleware, tuple):
                middleware, kwargs = middleware
                self.app.add_middleware(middleware, **kwargs)
            elif callable(middleware):
                self.app.add_middleware(middleware)

    @staticmethod
    def generate_client_file():
        src_path = os.path.join(os.path.dirname(__file__), "python_client.py")
        dest_path = os.path.join(os.getcwd(), "client.py")

        if os.path.exists(dest_path):
            return

        # Copy the file to the destination directory
        try:
            shutil.copy(src_path, dest_path)
            print(f"File '{src_path}' copied to '{dest_path}'")
        except Exception as e:
            print(f"Error copying file: {e}")

    def run(
        self,
        port: Union[str, int] = 8000,
        num_api_servers: Optional[int] = None,
        log_level: str = "info",
        generate_client_file: bool = True,
        api_server_worker_type: Optional[str] = None,
        **kwargs,
    ):
        if generate_client_file:
            LitServer.generate_client_file()

        port_msg = f"port must be a value from 1024 to 65535 but got {port}"
        try:
            port = int(port)
        except ValueError:
            raise ValueError(port_msg)

        if not (1024 <= port <= 65535):
            raise ValueError(port_msg)

        config = uvicorn.Config(app=self.app, host="0.0.0.0", port=port, log_level=log_level, **kwargs)
        sockets = [config.bind_socket()]

        if num_api_servers is None:
            num_api_servers = len(self.inference_workers)

        if num_api_servers < 1:
            raise ValueError("num_api_servers must be greater than 0")

        if sys.platform == "win32":
            warnings.warn(
                "Windows does not support forking. Using threads" " api_server_worker_type will be set to 'thread'"
            )
            api_server_worker_type = "thread"
        elif api_server_worker_type is None:
            api_server_worker_type = "process"

        manager, litserve_workers = self.launch_inference_worker(num_api_servers)

        try:
            servers = self._start_server(port, num_api_servers, log_level, sockets, api_server_worker_type, **kwargs)
            print(f"Swagger UI is available at http://0.0.0.0:{port}/docs")
            for s in servers:
                s.join()
        finally:
            print("Shutting down LitServe")
            for w in litserve_workers:
                w.terminate()
                w.join()
            manager.shutdown()

    def _start_server(self, port, num_uvicorn_servers, log_level, sockets, uvicorn_worker_type, **kwargs):
        servers = []
        for response_queue_id in range(num_uvicorn_servers):
            self.app.response_queue_id = response_queue_id
            if self.lit_spec:
                self.lit_spec.response_queue_id = response_queue_id
            app = copy.copy(self.app)

            config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level=log_level, **kwargs)
            server = uvicorn.Server(config=config)
            if uvicorn_worker_type == "process":
                ctx = mp.get_context("fork")
                w = ctx.Process(target=server.run, args=(sockets,))
            elif uvicorn_worker_type == "thread":
                w = threading.Thread(target=server.run, args=(sockets,))
            else:
                raise ValueError("Invalid value for api_server_worker_type. Must be 'process' or 'thread'")
            w.start()
            servers.append(w)
        return servers

    def setup_auth(self):
        if hasattr(self.lit_api, "authorize") and callable(self.lit_api.authorize):
            return self.lit_api.authorize
        if LIT_SERVER_API_KEY:
            return api_key_auth
        return no_auth
