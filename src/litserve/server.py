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
import pickle
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence, Tuple, Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.gzip import GZipMiddleware
from starlette.formparsers import MultiPartParser
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litserve import LitAPI
from litserve.connector import _Connector
from litserve.specs import OpenAISpec
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, load_and_raise
from collections import deque
import uvloop

mp.allow_connection_pickling()
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)

# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")

# timeout when we need to poll or wait indefinitely for a result in a loop.
LONG_TIMEOUT = 100

# FastAPI writes form files to disk over 1MB by default, which prevents serialization by multiprocessing
MultiPartParser.max_file_size = sys.maxsize


def _inject_context(context: Union[List[dict], dict], func, *args, **kwargs):
    sig = inspect.signature(func)
    if "context" in sig.parameters:
        return func(*args, **kwargs, context=context)
    return func(*args, **kwargs)


def get_batch_from_uid(uids, lit_api, request_buffer):
    batches = []
    for uid in uids:
        try:
            x_enc, pipe_s = request_buffer.pop(uid)
        except KeyError:
            continue
        batches.append((x_enc, pipe_s))
    return batches


def collate_requests(
    lit_api: LitAPI, request_queue: Queue, max_batch_size: int, batch_timeout: float
) -> Tuple[List, List]:
    payloads = []
    timed_out_uids = []
    entered_at = time.monotonic()
    end_time = entered_at + batch_timeout
    apply_timeout = lit_api.request_timeout not in (-1, False)

    while time.monotonic() < end_time and len(payloads) < max_batch_size:
        remaining_time = end_time - time.monotonic()
        if remaining_time <= 0:
            break

        try:
            uid, timestamp, x_enc = request_queue.get(timeout=min(remaining_time, 0.001))
            if apply_timeout and time.monotonic() - timestamp > lit_api.request_timeout:
                timed_out_uids.append(uid)
            else:
                payloads.append((uid, x_enc))

        except Empty:
            continue

    return payloads, timed_out_uids


def run_single_loop(lit_api: LitAPI, lit_spec: LitSpec, request_queue: Queue, response_queue: Queue):
    while True:
        try:
            uid, timestamp, x_enc = request_queue.get(timeout=1.0)
        except (Empty, ValueError):
            continue

        if (lit_api.request_timeout and lit_api.request_timeout != -1) and (
            time.monotonic() - timestamp > lit_api.request_timeout
        ):
            logger.error(
                f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                "has been timed out. "
                "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
            )
            response_queue.put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))
            continue
        try:
            context = {}
            if hasattr(lit_spec, "populate_context"):
                lit_spec.populate_context(context, x_enc)
            x = _inject_context(
                context,
                lit_api.decode_request,
                x_enc,
            )
            y = _inject_context(
                context,
                lit_api.predict,
                x,
            )
            y_enc = _inject_context(
                context,
                lit_api.encode_response,
                y,
            )
            response_queue.put((uid, (y_enc, LitAPIStatus.OK)))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            err_pkl = pickle.dumps(e)
            response_queue.put((uid, (err_pkl, LitAPIStatus.ERROR)))


def run_batched_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    response_queue: Queue,
    max_batch_size: int,
    batch_timeout: float,
):
    while True:
        batches, timed_out_uids = collate_requests(
            lit_api,
            request_queue,
            max_batch_size,
            batch_timeout,
        )

        for uid in timed_out_uids:
            logger.error(
                f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                "has been timed out. "
                "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
            )
            response_queue.put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))

        if not batches:
            continue
        logger.debug(f"{len(batches)} batched requests received")
        uids, inputs = zip(*batches)
        try:
            contexts = [{}] * len(inputs)
            if hasattr(lit_spec, "populate_context"):
                for input, context in zip(inputs, contexts):
                    lit_spec.populate_context(context, input)

            x = [
                _inject_context(
                    context,
                    lit_api.decode_request,
                    input,
                )
                for input, context in zip(inputs, contexts)
            ]
            x = lit_api.batch(x)
            y = _inject_context(contexts, lit_api.predict, x)
            outputs = lit_api.unbatch(y)
            for y, uid, context in zip(outputs, uids, contexts):
                y_enc = _inject_context(context, lit_api.encode_response, y)

                response_queue.put((uid, (y_enc, LitAPIStatus.OK)))

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the batched request.\n"
                "Please check the error trace for more details."
            )
            err_pkl = pickle.dumps(e)
            for uid in uids:
                response_queue.put((uid, (err_pkl, LitAPIStatus.ERROR)))


def run_streaming_loop(lit_api: LitAPI, lit_spec: LitSpec, request_queue: Queue, response_queue: Queue):
    while True:
        try:
            uid, timestamp, x_enc = request_queue.get(timeout=1.0)
            logger.debug("uid=%s", uid)
        except (Empty, ValueError):
            continue

        if (lit_api.request_timeout and lit_api.request_timeout != -1) and (
            time.monotonic() - timestamp > lit_api.request_timeout
        ):
            logger.error(
                f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                "has been timed out. "
                "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
            )
            response_queue.put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))
            continue

        try:
            context = {}
            if hasattr(lit_spec, "populate_context"):
                lit_spec.populate_context(context, x_enc)
            x = _inject_context(
                context,
                lit_api.decode_request,
                x_enc,
            )
            y_gen = _inject_context(
                context,
                lit_api.predict,
                x,
            )
            y_enc_gen = _inject_context(
                context,
                lit_api.encode_response,
                y_gen,
            )
            for y_enc in y_enc_gen:
                y_enc = lit_api.format_encoded_response(y_enc)
                response_queue.put((uid, (y_enc, LitAPIStatus.OK)))
            response_queue.put((uid, ("", LitAPIStatus.FINISH_STREAMING)))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            response_queue.put((uid, (pickle.dumps(e), LitAPIStatus.ERROR)))


def run_batched_streaming_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    response_queue: Queue,
    max_batch_size: int,
    batch_timeout: float,
):
    while True:
        batches, timed_out_uids = collate_requests(
            lit_api,
            request_queue,
            max_batch_size,
            batch_timeout,
        )
        for uid in timed_out_uids:
            logger.error(
                f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                "has been timed out. "
                "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
            )
            response_queue.put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))

        if not batches:
            continue
        uids, inputs = zip(*batches)
        try:
            contexts = [{}] * len(inputs)
            if hasattr(lit_spec, "populate_context"):
                for input, context in zip(inputs, contexts):
                    lit_spec.populate_context(context, input)

            x = [
                _inject_context(
                    context,
                    lit_api.decode_request,
                    input,
                )
                for input, context in zip(inputs, contexts)
            ]
            x = lit_api.batch(x)
            y_iter = _inject_context(contexts, lit_api.predict, x)
            unbatched_iter = lit_api.unbatch(y_iter)
            y_enc_iter = _inject_context(contexts, lit_api.encode_response, unbatched_iter)

            # y_enc_iter -> [[response-1, response-2], [response-1, response-2]]
            for y_batch in y_enc_iter:
                for y_enc, uid in zip(y_batch, uids):
                    y_enc = lit_api.format_encoded_response(y_enc)
                    response_queue.put((uid, (y_enc, LitAPIStatus.OK)))

            for uid in uids:
                response_queue.put((uid, ("", LitAPIStatus.FINISH_STREAMING)))

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming batched request.\n"
                "Please check the error trace for more details."
            )
            err_pkl = pickle.dumps(e)
            response_queue.put((uid, (err_pkl, LitAPIStatus.ERROR)))


def inference_worker(
    lit_api: LitAPI,
    lit_spec: Optional[LitSpec],
    device: str,
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
    max_batch_size: int,
    batch_timeout: float,
    stream: bool,
    workers_setup_status: Dict[str, bool] = None,
):
    
    lit_api.setup(device)
    lit_api.device = device

    print(f"Setup complete for worker {worker_id}.")

    config = workers_setup_status["config"]
    sockets = workers_setup_status["sockets"]
    lit_server, th = create_server(lit_api, lit_spec, config, sockets, )  # inits a new FastAPI instance for uvicorn
    lit_server.response_queue = response_queue
    lit_server.request_queue = request_queue
    lit_server.workers_setup_status = workers_setup_status
    th.start()

    if workers_setup_status:
        workers_setup_status[worker_id] = True

    if lit_spec:
        logging.info(f"LitServe will use {lit_spec.__class__.__name__} spec")
    if stream:
        if max_batch_size > 1:
            run_batched_streaming_loop(lit_api, lit_spec, request_queue, response_queue, max_batch_size, batch_timeout)
        else:
            run_streaming_loop(lit_api, lit_spec, request_queue, response_queue)
        return

    if max_batch_size > 1:
        run_batched_loop(lit_api, lit_spec, request_queue, response_queue, max_batch_size, batch_timeout)
    else:
        run_single_loop(
            lit_api,
            lit_spec,
            request_queue,
            response_queue,
        )


def no_auth():
    pass


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


async def response_queue_to_buffer(
    response_queue: mp.Queue,
    buffer: Dict[str, Union[Tuple[deque, asyncio.Event], asyncio.Event]],
    stream: bool,
    response_executor: ThreadPoolExecutor,
):
    loop = asyncio.get_running_loop()
    if stream:
        while True:
            try:
                uid, payload = await loop.run_in_executor(response_executor, response_queue.get)
            except Empty:
                await asyncio.sleep(0.0001)
                continue
            q, event = buffer[uid]
            q.append(payload)
            event.set()

    else:
        while True:
            uid, payload = await loop.run_in_executor(response_executor, response_queue.get)
            event = buffer.pop(uid)
            buffer[uid] = payload
            event.set()

def create_server(lit_api, lit_spec, config, sockets, **kwargs):
    lit_server = LitServer(lit_api=lit_api, spec=lit_spec)
    config.app = lit_server.app
    server = uvicorn.Server(config=config)
    th = threading.Thread(target=server.run, args=(sockets,), daemon=True)
    return lit_server, th

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
    ):
        self.litserve_locals = locals()
        if batch_timeout > timeout and timeout not in (False, -1):
            raise ValueError("batch_timeout must be less than timeout")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")
        if isinstance(spec, OpenAISpec):
            stream = True

        if not api_path.startswith("/"):
            raise ValueError(
                "api_path must start with '/'. "
                "Please provide a valid api path like '/predict', '/classify', or '/v1/predict'"
            )

        self.api_path = api_path
        lit_api.stream = stream
        lit_api.request_timeout = timeout
        lit_api.sanitize(max_batch_size, spec=spec)
        self.app = FastAPI(lifespan=self.lifespan)
        # gzip does not play nicely with streaming, see https://github.com/tiangolo/fastapi/discussions/8448
        if not stream:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        if max_payload_size is not None:
            self.app.add_middleware(MaxSizeMiddleware, max_size=max_payload_size)
        self.lit_api = lit_api
        self.lit_spec = spec
        self.workers_per_device = workers_per_device
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.batch_timeout = batch_timeout
        self.stream = stream
        self._connector = _Connector(accelerator=accelerator, devices=devices)

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

        self.workers = self.devices * self.workers_per_device
        self.setup_server()


    def launch_inference_worker(self, manager, config, sockets):
        self.workers_setup_status = manager.dict()
        self.response_queues = []
        self.process_list = []
        self.request_queues = []

        config = uvicorn.Config(app=None, port=8000, log_level="info")
        self.workers_setup_status["config"] = config
        self.workers_setup_status["sockets"] = sockets

        for worker_id, device in enumerate(self.devices * self.workers_per_device):
            if len(device) == 1:
                device = device[0]

            self.workers_setup_status[worker_id] = False
            request_queue = manager.Queue()
            self.request_queues.append(request_queue)
            response_queue = manager.Queue()
            self.response_queues.append(response_queue)

            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=inference_worker,
                args=(
                    self.lit_api,
                    self.lit_spec,
                    device,
                    worker_id,
                    request_queue,
                    response_queue,
                    self.max_batch_size,
                    self.batch_timeout,
                    self.stream,
                    self.workers_setup_status,
                ),
                daemon=True,
            )
            process.start()
            self.process_list.append((process, worker_id))

        for spec in self._specs:
            # Objects of Server class are referenced (not copied)
            logging.debug(f"shallow copy for Server is created for for spec {spec}")
            server_copy = copy.copy(self)
            del server_copy.app
            try:
                spec.setup(server_copy)
            except Exception as e:
                raise e
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        loop = asyncio.get_running_loop()
        self.response_buffer = dict()

        response_executor = ThreadPoolExecutor(max_workers=len(self.devices * self.workers_per_device))
        future = response_queue_to_buffer(self.response_queue, self.response_buffer, self.stream, response_executor)
        task = loop.create_task(future)

        yield

        task.cancel()


    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    async def data_streamer(self, q: deque, data_available: asyncio.Event, send_status: bool = False):
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

    def setup_server(self):
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

        async def predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
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

            self.request_queue.put_nowait((uid, time.monotonic(), payload))

            await event.wait()
            response, status = self.response_buffer.pop(uid)

            if status == LitAPIStatus.ERROR:
                load_and_raise(response)
            return response

        async def stream_predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
            uid = uuid.uuid4()
            event = asyncio.Event()
            q = deque()
            self.response_buffer[uid] = (q, event)
            logger.debug(f"Received request uid={uid}")

            payload = request
            if self.request_type == Request:
                payload = await request.json()
            self.request_queue.put((uid, time.monotonic(), payload))

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

    def generate_client_file(self):
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

    def run(self, port: Union[str, int] = 8000, log_level: str = "info", generate_client_file: bool = True, **kwargs):
        if generate_client_file:
            self.generate_client_file()

        manager = mp.Manager()

        port_msg = f"port must be a value from 1024 to 65535 but got {port}"
        try:
            port = int(port)
        except ValueError:
            raise ValueError(port_msg)

        if not (1024 <= port <= 65535):
            raise ValueError(port_msg)

        config = uvicorn.Config(app=self.app, port=port, log_level=log_level, loop="uvloop")
        sockets = [config.bind_socket()]

        self.launch_inference_worker(manager, config, sockets)
        for p, worker_id in self.process_list:
            p.join()

    def setup_auth(self):
        if hasattr(self.lit_api, "authorize") and callable(self.lit_api.authorize):
            return self.lit_api.authorize
        if LIT_SERVER_API_KEY:
            return api_key_auth
        return no_auth


class MaxSizeMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_size: Optional[int] = None,
    ) -> None:
        self.app = app
        self.max_size = max_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        total_size = 0

        async def rcv() -> Message:
            nonlocal total_size
            message = await receive()
            chunk_size = len(message.get("body", b""))
            total_size += chunk_size
            if self.max_size is not None and total_size > self.max_size:
                raise HTTPException(413, "Payload too large")
            return message

        await self.app(scope, rcv, send)
