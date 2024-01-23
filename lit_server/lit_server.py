import asyncio
from contextlib import asynccontextmanager
import inspect
from multiprocessing import Process, Manager
import os
import shutil
import time
from typing import Sequence
import uuid

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.security import APIKeyHeader

from lit_server import LitAPI


# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")


def inference_worker(lit_api, device, worker_id, request_buffer, response_buffer):
    lit_api.setup(device=device)

    while True:
        # TODO: we can easily implement batching here: just keep getting
        #       items from the buffer, fill a batch, predict, assign outputs
        #       to the buffer
        #       We could also expose the batching strategy at the LitAPI level
        try:
            uid = next(iter(request_buffer.keys()))
            x = request_buffer.pop(uid)
        except (StopIteration, KeyError):
            time.sleep(0.05)
            continue

        x = lit_api.decode_request(x)

        y = lit_api.predict(x)

        # response_buffer[uid] = y
        response_buffer[uid] = lit_api.encode_response(y)


def no_auth():
    pass


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name='X-API-Key'))):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


def setup_auth():
    if LIT_SERVER_API_KEY:
        return api_key_auth
    return no_auth


def cleanup(request_buffer, uid):
    try:
        request_buffer.pop(uid)
    except KeyError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = Manager()
    app.request_buffer = manager.dict()
    app.response_buffer = manager.dict()

    # NOTE: device: str | List[str], the latter in the case a model needs more than one device to run
    for worker_id, device in enumerate(app.devices * app.workers_per_device):
        if len(device) == 1:
            device = device[0]
        process = Process(
            target=inference_worker,
            args=(app.lit_api, device, worker_id, app.request_buffer, app.response_buffer),
            daemon=True)
        process.start()

    yield


class LitServer:
    # TODO: add support for accelerator="auto", devices="auto"
    def __init__(self, lit_api: LitAPI, accelerator="cpu", devices=1, workers_per_device=1):
        self.app = FastAPI(lifespan=lifespan)
        self.app.lit_api = lit_api
        self.app.workers_per_device = workers_per_device

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
        # @self.app.post("/predict", dependencies=[Depends(setup_auth())])
        @self.app.post("/predict")
        async def predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
            uid = uuid.uuid4()

            if self.request_type == Request:
                self.app.request_buffer[uid] = await request.json()
            else:
                self.app.request_buffer[uid] = request
            background_tasks.add_task(cleanup, self.app.request_buffer, uid)

            output = None

            while True:
                await asyncio.sleep(0.05)
                if uid in self.app.response_buffer:
                    output = self.app.response_buffer.pop(uid)
                    break

            return output

    def generate_client_file(self):
        src_path = os.path.join(os.path.dirname(__file__), "python_client.py")
        dest_path = os.path.join(os.getcwd(), 'client.py')

        # Copy the file to the destination directory
        try:
            shutil.copy(src_path, dest_path)
            print(f"File '{src_path}' copied to '{dest_path}'")
        except Exception as e:
            print(f"Error copying file: {e}")

    def run(self, port=8000, timeout_keep_alive=30):
        self.generate_client_file()

        import uvicorn
        uvicorn.run(host="0.0.0.0", port=port, app=self.app, timeout_keep_alive=timeout_keep_alive, log_level="info")
