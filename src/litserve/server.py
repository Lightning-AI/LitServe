# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import asyncio
import contextlib
from contextlib import asynccontextmanager
import inspect
from multiprocessing import Process, Manager, Queue, Pipe
from queue import Empty
import time
import os
import shutil
from typing import Sequence
import uuid

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.security import APIKeyHeader
import sys
from litserve import LitAPI

# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")


def get_batch_from_uid(uids, lit_api, request_buffer):
    batches = []
    for uid in uids:
        try:
            x_enc, pipe_s = request_buffer.pop(uid)
        except KeyError:
            continue
        x = lit_api.decode_request(x_enc)
        batches.append((x, pipe_s))
    return batches


def collate_requests(lit_api, request_queue: Queue, request_buffer, max_batch_size, batch_timeout):
    uids = []
    entered_at = time.time()
    while (batch_timeout - (time.time() - entered_at) > 0) and len(uids) <= max_batch_size:
        try:
            uid = request_queue.get_nowait()
            uids.append(uid)
        except (Empty, ValueError):
            continue
    return get_batch_from_uid(uids, lit_api, request_buffer)


def run_batched_loop(lit_api, request_queue: Queue, request_buffer, max_batch_size, batch_timeout):
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

        x = lit_api.batch(inputs)
        y = lit_api.predict(x)
        outputs = lit_api.unbatch(y)

        for y, pipe_s in zip(outputs, pipes):
            y_enc = lit_api.encode_response(y)

            with contextlib.suppress(BrokenPipeError):
                pipe_s.send(y_enc)


def run_single_loop(lit_api, request_queue: Queue, request_buffer):
    while True:
        try:
            uid = request_queue.get(timeout=0.01)
            try:
                x_enc, pipe_s = request_buffer.pop(uid)
            except KeyError:
                continue
        except (Empty, ValueError):
            continue

        x = lit_api.decode_request(x_enc)
        y = lit_api.predict(x)
        y_enc = lit_api.encode_response(y)
        with contextlib.suppress(BrokenPipeError):
            pipe_s.send(y_enc)


def inference_worker(
        lit_api,
        device,
        worker_id,
        request_queue,
        request_buffer,
        max_batch_size,
        batch_timeout,
):
    lit_api.setup(device=device)
    if max_batch_size > 1:
        run_batched_loop(lit_api, request_queue, request_buffer, max_batch_size, batch_timeout)
    else:
        run_single_loop(lit_api, request_queue, request_buffer,)


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
    with contextlib.suppress(KeyError):
        request_buffer.pop(uid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.request_queue = Queue()
    manager = Manager()
    app.request_buffer = manager.dict()

    # NOTE: device: str | List[str], the latter in the case a model needs more than one device to run
    for worker_id, device in enumerate(app.devices * app.workers_per_device):
        if len(device) == 1:
            device = device[0]
        process = Process(
            target=inference_worker,
            args=(
                app.lit_api,
                device,
                worker_id,
                app.request_queue,
                app.request_buffer,
                app.max_batch_size,
                app.batch_timeout,
            ),
            daemon=True,
        )
        process.start()

    yield


class LitServer:
    # TODO: add support for accelerator="auto", devices="auto"
    def __init__(
            self,
            lit_api: LitAPI,
            accelerator="cpu",
            devices=1,
            workers_per_device=1,
            timeout=30,
            max_batch_size=1,
            batch_timeout=1.0,
    ):
        if batch_timeout > timeout:
            raise ValueError("batch_timeout must be less than timeout")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")

        self.app = FastAPI(lifespan=lifespan)
        self.app.lit_api = lit_api
        self.app.workers_per_device = workers_per_device
        self.app.timeout = timeout
        self.app.max_batch_size = max_batch_size
        self.app.batch_timeout = batch_timeout
        initial_pool_size = 100
        self.max_pool_size = 1000
        self.pipe_pool = [Pipe() for _ in range(initial_pool_size)]

        decode_request_signature = inspect.signature(lit_api.decode_request)
        encode_response_signature = inspect.signature(lit_api.encode_response)

        self.request_type = decode_request_signature.parameters["request"].annotation
        if self.request_type == decode_request_signature.empty:
            self.request_type = Request

        self.response_type = encode_response_signature.return_annotation
        if self.response_type == encode_response_signature.empty:
            self.response_type = Response

        if accelerator == "cpu":
            self.app.devices = [accelerator]
        elif accelerator in ["cuda", "gpu"]:
            device_list = devices
            if isinstance(devices, int):
                device_list = range(devices)
            self.app.devices = [self.device_identifiers(accelerator, device) for device in device_list]

        self.setup_server()

    def new_pipe(self):
        try:
            pipe_s, pipe_r = self.pipe_pool.pop()
        except IndexError:
            pipe_s, pipe_r = Pipe()
        return pipe_s, pipe_r

    def dispose_pipe(self, pipe_s, pipe_r):
        if len(self.pipe_pool) > self.max_pool_size:
            return
        self.pipe_pool.append(pipe_s, pipe_r)

    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    def setup_server(self):
        @self.app.get("/", dependencies=[Depends(setup_auth())])
        async def index(request: Request) -> Response:
            return Response(content="litserve running")

        @self.app.post("/predict", dependencies=[Depends(setup_auth())])
        async def predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
            uid = uuid.uuid4()

            read, write = self.new_pipe()

            if self.request_type == Request:
                self.app.request_buffer[uid] = (await request.json(), write)
            else:
                self.app.request_buffer[uid] = (request, write)

            self.app.request_queue.put(uid)
            background_tasks.add_task(cleanup, self.app.request_buffer, uid)

            async def event_wait(evt, timeout):
                # suppress TimeoutError because we'll return False in case of timeout
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(evt.wait(), timeout)
                return evt.is_set()

            def get_from_pipe():
                if read.poll(self.app.timeout):
                    return read.recv()
                return HTTPException(status_code=504, detail="Request timed out")

            async def data_reader():
                data_available = asyncio.Event()
                asyncio.get_event_loop().add_reader(read.fileno(), data_available.set)

                if not read.poll():
                    await event_wait(data_available, self.app.timeout)
                    data_available.clear()
                asyncio.get_event_loop().remove_reader(read.fileno())

                if read.poll():
                    return read.recv()
                return HTTPException(status_code=504, detail="Request timed out")

            if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
                data = await asyncio.to_thread(get_from_pipe)
            else:
                data = await data_reader()

            if type(data) == HTTPException:
                raise data

            return data

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

    def run(self, port=8000, log_level="info", **kwargs):
        self.generate_client_file()

        import uvicorn

        uvicorn.run(host="0.0.0.0", port=port, app=self.app, log_level=log_level, **kwargs)
