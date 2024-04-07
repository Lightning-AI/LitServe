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

from litserve import LitAPI


# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")


def inference_worker(lit_api, device, worker_id, request_queue, request_buffer, max_batch_size, batch_timeout):
    lit_api.setup(device=device)
    while True:
        # NOTE: to implement batching here: keep getting items from the queue,
        #       fill a batch, predict, send outputs to the respective pipes
        #       In the future we will expose this through the API.
        batches = []
        start_time = time.time()
        while len(batches) <= max_batch_size and time.time() - start_time < batch_timeout:
            try:
                uid = request_queue.get(timeout=batch_timeout)
                try:
                    x_enc, pipe_s = request_buffer.pop(uid)
                except KeyError:
                    continue
                x = lit_api.decode_request(x_enc)
                batches.append((x, pipe_s))
            except (Empty, ValueError):
                continue

        if not batches:
            continue
        x_batch, pipe_s_batch = zip(*batches)
        y_batch = lit_api.predict(x_batch)

        for y, pipe_s in zip(y_batch, pipe_s_batch):
            y_enc = lit_api.encode_response(y)
            with contextlib.suppress(BrokenPipeError):
                pipe_s.send(y_enc)


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
        self.app = FastAPI(lifespan=lifespan)
        self.app.lit_api = lit_api
        self.app.workers_per_device = workers_per_device
        self.app.timeout = timeout
        self.app.max_batch_size = max_batch_size
        self.app.batch_timeout = batch_timeout

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

            pipe_s, pipe_r = Pipe()

            if self.request_type == Request:
                self.app.request_buffer[uid] = (await request.json(), pipe_s)
            else:
                self.app.request_buffer[uid] = (request, pipe_s)

            self.app.request_queue.put(uid)

            background_tasks.add_task(cleanup, self.app.request_buffer, uid)

            async def get_from_pipe():
                if pipe_r.poll(self.app.timeout):
                    return pipe_r.recv()
                return HTTPException(status_code=504, detail="Request timed out")

            data = await asyncio.get_running_loop().create_task(get_from_pipe())

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
