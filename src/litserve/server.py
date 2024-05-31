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
import logging
import pickle
from contextlib import asynccontextmanager
import inspect
import multiprocessing as mp
from multiprocessing import Manager, Queue, Pipe
from multiprocessing.connection import Connection
from queue import Empty
import uvicorn
import time
import os
import shutil
from typing import Sequence, Optional, Union
import uuid

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.security import APIKeyHeader
import sys

from fastapi.responses import StreamingResponse

from litserve import LitAPI
from litserve.connector import _Connector
from litserve.specs import OpenAISpec
from litserve.specs.base import LitSpec
from litserve.utils import wait_for_queue_timeout, LitAPIStatus, load_and_raise

logger = logging.getLogger(__name__)

# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")

# timeout when we need to poll or wait indefinitely for a result in a loop.
LONG_TIMEOUT = 100


def _inject_context(context: dict, func, *args, **kwargs):
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


def collate_requests(lit_api, request_queue: Queue, request_buffer, max_batch_size, batch_timeout):
    uids = []
    entered_at = time.time()
    while (batch_timeout - (time.time() - entered_at) > 0) and len(uids) < max_batch_size:
        try:
            uid = request_queue.get(timeout=0.001)
            uids.append(uid)
        except (Empty, ValueError):
            continue
    return get_batch_from_uid(uids, lit_api, request_buffer)


def run_batched_loop(lit_api, lit_spec, request_queue: Queue, request_buffer, max_batch_size, batch_timeout):
    while True:
        batches = collate_requests(
            lit_api,
            request_queue,
            request_buffer,
            max_batch_size,
            batch_timeout,
        )
        if not batches:
            continue

        logger.debug(f"{len(batches)} batched requests received")
        inputs, pipes = zip(*batches)

        try:
            x = [lit_api.decode_request(input) for input in inputs]
            x = lit_api.batch(x)
            y = lit_api.predict(x)
            outputs = lit_api.unbatch(y)
            for y, pipe_s in zip(outputs, pipes):
                y_enc = lit_api.encode_response(y)

                with contextlib.suppress(BrokenPipeError):
                    pipe_s.send((y_enc, LitAPIStatus.OK))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the batched request.\n"
                "Please check the error trace for more details."
            )
            err_pkl = pickle.dumps(e)
            with contextlib.suppress(BrokenPipeError):
                for pipe_s in pipes:
                    pipe_s.send((err_pkl, LitAPIStatus.ERROR))


def run_single_loop(lit_api, lit_spec, request_queue: Queue, request_buffer):
    while True:
        try:
            uid = request_queue.get(timeout=1.0)
            try:
                x_enc, pipe_s = request_buffer.pop(uid)
            except KeyError:
                continue
        except (Empty, ValueError):
            continue
        try:
            context = {}
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

            with contextlib.suppress(BrokenPipeError):
                pipe_s.send((y_enc, LitAPIStatus.OK))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            with contextlib.suppress(BrokenPipeError):
                pipe_s.send((pickle.dumps(e), LitAPIStatus.ERROR))


def run_streaming_loop(lit_api: LitAPI, lit_spec, request_queue: Queue, request_buffer):
    while True:
        try:
            uid = request_queue.get(timeout=1.0)
            logger.debug("uid=%s", uid)
            try:
                x_enc, pipe_s = request_buffer.pop(uid)
            except KeyError:
                continue
        except (Empty, ValueError):
            continue

        try:
            context = {}
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
                with contextlib.suppress(BrokenPipeError):
                    y_enc = lit_api.format_encoded_response(y_enc)
                    pipe_s.send((y_enc, LitAPIStatus.OK))
            with contextlib.suppress(BrokenPipeError):
                pipe_s.send(("", LitAPIStatus.FINISH_STREAMING))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            with contextlib.suppress(BrokenPipeError):
                pipe_s.send((pickle.dumps(e), LitAPIStatus.ERROR))


def run_batched_streaming_loop(lit_api, lit_spec, request_queue: Queue, request_buffer, max_batch_size, batch_timeout):
    while True:
        batches = collate_requests(
            lit_api,
            request_queue,
            request_buffer,
            max_batch_size,
            batch_timeout,
        )
        if not batches:
            continue

        inputs, pipes = zip(*batches)

        try:
            x = [lit_api.decode_request(input) for input in inputs]
            x = lit_api.batch(x)
            y_iter = lit_api.predict(x)
            unbatched_iter = lit_api.unbatch(y_iter)
            y_enc_iter = lit_api.encode_response(unbatched_iter)

            # y_enc_iter -> [[response-1, response-2], [response-1, response-2]]
            for y_batch in y_enc_iter:
                for y_enc, pipe_s in zip(y_batch, pipes):
                    with contextlib.suppress(BrokenPipeError):
                        y_enc = lit_api.format_encoded_response(y_enc)
                        pipe_s.send((y_enc, LitAPIStatus.OK))

            with contextlib.suppress(BrokenPipeError):
                for pipe_s in pipes:
                    pipe_s.send(("", LitAPIStatus.FINISH_STREAMING))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming batched request.\n"
                "Please check the error trace for more details."
            )
            err = pickle.dumps(e)
            for pipe_s in pipes:
                pipe_s.send((err, LitAPIStatus.ERROR))


def inference_worker(
    lit_api: LitAPI,
    lit_spec: Optional[LitSpec],
    device,
    worker_id,
    request_queue,
    request_buffer,
    max_batch_size,
    batch_timeout,
    stream,
):
    lit_api.setup(device)
    lit_api.device = device
    message = f"Setup complete for worker {worker_id}."
    print(message)
    logger.info(message)
    if lit_spec:
        logging.info(f"LitServe will use {lit_spec.__class__.__name__} spec")
    if stream:
        if max_batch_size > 1:
            run_batched_streaming_loop(lit_api, lit_spec, request_queue, request_buffer, max_batch_size, batch_timeout)
        else:
            run_streaming_loop(lit_api, lit_spec, request_queue, request_buffer)
        return

    if max_batch_size > 1:
        run_batched_loop(lit_api, lit_spec, request_queue, request_buffer, max_batch_size, batch_timeout)
    else:
        run_single_loop(
            lit_api,
            lit_spec,
            request_queue,
            request_buffer,
        )


def no_auth():
    pass


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


def setup_auth():
    if LIT_SERVER_API_KEY:
        return api_key_auth
    return no_auth


def cleanup(request_buffer, uid):
    logger.debug("Cleaning up request uid=%s", uid)
    with contextlib.suppress(KeyError):
        request_buffer.pop(uid)


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
        stream: bool = False,
        spec: Optional[LitSpec] = None,
    ):
        if batch_timeout > timeout and timeout not in (False, -1):
            raise ValueError("batch_timeout must be less than timeout")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")
        if isinstance(spec, OpenAISpec):
            stream = True

        lit_api.stream = stream
        lit_api.sanitize(max_batch_size, spec=spec)
        self.app = FastAPI(lifespan=self.lifespan)
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

        self.setup_server()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        manager = Manager()
        self.request_buffer = manager.dict()
        self.request_queue = manager.Queue()

        try:
            pickle.dumps(self.lit_api)
            pickle.dumps(self.lit_spec)

        except (pickle.PickleError, AttributeError) as e:
            logging.error(
                "The LitAPI instance provided to LitServer cannot be moved to a worker because "
                "it cannot be pickled. Please ensure all heavy-weight operations, like model "
                "creation, are defined in LitAPI's setup method."
            )
            raise e

        process_list = []
        # NOTE: device: str | List[str], the latter in the case a model needs more than one device to run
        for worker_id, device in enumerate(self.devices * self.workers_per_device):
            if len(device) == 1:
                device = device[0]

            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=inference_worker,
                args=(
                    self.lit_api,
                    self.lit_spec,
                    device,
                    worker_id,
                    self.request_queue,
                    self.request_buffer,
                    self.max_batch_size,
                    self.batch_timeout,
                    self.stream,
                ),
                daemon=True,
            )
            process.start()
            process_list.append((process, worker_id))

        for spec in self._specs:
            # Objects of Server class are referenced (not copied)
            logging.debug(f"shallow copy for Server is created for for spec {spec}")
            server_copy = copy.copy(self)
            del server_copy.app
            spec.setup(server_copy)

        yield

        for process, worker_id in process_list:
            logging.info(f"terminating worker worker_id={worker_id}")
            process.terminate()

    def new_pipe(self) -> tuple:
        return Pipe()

    def close_pipe(self, pipe_s, pipe_r):
        pipe_s.close()
        pipe_r.close()

    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    def get_from_pipe(self, read):
        while True:
            if read.poll(LONG_TIMEOUT):
                return read.recv()

    async def data_reader(self, read):
        data_available = asyncio.Event()
        asyncio.get_event_loop().add_reader(read.fileno(), data_available.set)

        if not read.poll():
            await data_available.wait()
        data_available.clear()
        asyncio.get_event_loop().remove_reader(read.fileno())
        return read.recv()

    async def win_data_streamer(self, read, write):
        # this is a workaround for Windows since asyncio loop.add_reader is not supported.
        # https://docs.python.org/3/library/asyncio-platforms.html
        while True:
            if read.poll(LONG_TIMEOUT):
                response, status = read.recv()
                if status == LitAPIStatus.FINISH_STREAMING:
                    self.close_pipe(read, write)
                    return
                elif status == LitAPIStatus.ERROR:
                    self.close_pipe(read, write)
                    logger.error(
                        "Error occurred while streaming outputs from the inference worker. "
                        "Please check the above traceback."
                    )
                    return
                yield response

            await asyncio.sleep(0.0001)

    async def data_streamer(self, read: Connection, write: Connection):
        data_available = asyncio.Event()
        while True:
            if not read.poll(1):
                asyncio.get_event_loop().add_reader(read.fileno(), data_available.set)
                await data_available.wait()
                data_available.clear()
                asyncio.get_event_loop().remove_reader(read.fileno())
            if read.poll(1):
                response, status = read.recv()
                if status == LitAPIStatus.FINISH_STREAMING:
                    return
                if status == LitAPIStatus.ERROR:
                    logger.error(
                        "Error occurred while streaming outputs from the inference worker. "
                        "Please check the above traceback."
                    )
                    return
                yield response

    def cleanup_request(self, request_buffer, uid):
        with contextlib.suppress(KeyError):
            request_buffer.pop(uid)

    def setup_server(self):
        @self.app.get("/", dependencies=[Depends(setup_auth())])
        async def index(request: Request) -> Response:
            return Response(content="litserve running")

        async def predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
            uid = uuid.uuid4()
            logger.debug(f"Received request uid={uid}")

            read, write = self.new_pipe()

            if self.request_type == Request:
                if request.headers["Content-Type"] == "application/x-www-form-urlencoded" or request.headers[
                    "Content-Type"
                ].startswith("multipart/form-data"):
                    self.request_buffer[uid] = (await request.form(), write)
                else:
                    self.request_buffer[uid] = (await request.json(), write)
            else:
                self.request_buffer[uid] = (request, write)

            self.request_queue.put(uid)
            background_tasks.add_task(self.cleanup_request, self.request_buffer, uid)

            if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
                data = await wait_for_queue_timeout(
                    asyncio.to_thread(self.get_from_pipe, read), self.timeout, uid, self.request_buffer
                )
            else:
                data = await wait_for_queue_timeout(self.data_reader(read), self.timeout, uid, self.request_buffer)
            self.close_pipe(read, write)

            response, status = data
            if status == LitAPIStatus.ERROR:
                load_and_raise(response)
            return response

        async def stream_predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
            uid = uuid.uuid4()
            logger.debug(f"Received request uid={uid}")

            read, write = self.new_pipe()

            if self.request_type == Request:
                self.request_buffer[uid] = (await request.json(), write)
            else:
                self.request_buffer[uid] = (request, write)

            self.request_queue.put(uid)
            background_tasks.add_task(cleanup, self.request_buffer, uid)

            if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
                return StreamingResponse(self.win_data_streamer(read, write))

            return StreamingResponse(self.data_streamer(read, write))

        if not self._specs:
            stream = self.lit_api.stream
            # In the future we might want to differentiate endpoints for streaming vs non-streaming
            # For now we allow either one or the other
            endpoint = "/predict"
            methods = ["POST"]
            self.app.add_api_route(
                endpoint, stream_predict if stream else predict, methods=methods, dependencies=[Depends(setup_auth())]
            )

        for spec in self._specs:
            spec: LitSpec
            # TODO check that path is not clashing
            for path, endpoint, methods in spec.endpoints:
                self.app.add_api_route(path, endpoint=endpoint, methods=methods, dependencies=[Depends(setup_auth())])

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

    def run(self, port: Union[str, int] = 8000, log_level: str = "info", **kwargs):
        self.generate_client_file()

        port_msg = f"port must be a value from 1024 to 65535 but got {port}"
        try:
            port = int(port)
        except ValueError:
            raise ValueError(port_msg)

        if not (1024 <= port <= 65535):
            raise ValueError(port_msg)

        uvicorn.run(host="0.0.0.0", port=port, app=self.app, log_level=log_level, **kwargs)
