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
import logging
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import os
import pickle
import queue
import shutil
import signal
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from multiprocessing import Manager, Pipe, Queue
from multiprocessing import shared_memory, Lock
from multiprocessing.connection import Connection
from queue import Empty
from typing import Dict, List, Optional, Sequence, Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.gzip import GZipMiddleware

from litserve import LitAPI
from litserve.connector import _Connector
from litserve.specs import OpenAISpec
from litserve.specs.base import LitSpec
from litserve.utils import (
    LitAPIStatus,
    Timing,
    load_and_raise,
    log_time,
    pipe_read,
    pipe_send,
    server_logger,
    wait_for_queue_timeout,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler1 = logging.FileHandler("logs/app.log")
formatter1 = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler1.setFormatter(formatter1)
logger.addHandler(file_handler1)

BUFFER_SIZE = 10000

class LitSMQ:
    def __init__(self, name: str, metadata_shm: shared_memory.SharedMemory, data_shm: shared_memory.SharedMemory):
        self.data_size = data_shm.size
        self.name = name
        self.lock = Lock()

        self.metadata_shm = metadata_shm
        self.data_shm = data_shm
        self.metadata_buffer = metadata_shm.buf
        self.data_buffer = data_shm.buf

        self.head_index = 0
        self.tail_index = 4

    @staticmethod
    def create(name, data_size=10_000):
        try:
            metadata_shm = shared_memory.SharedMemory(create=True, size=128, name=name + '_metadata')
        except FileExistsError:
            metadata_shm = shared_memory.SharedMemory(name=name + '_metadata')
        try:
            data_shm = shared_memory.SharedMemory(create=True, size=data_size, name=name + '_data')
        except FileExistsError:
            data_shm = shared_memory.SharedMemory(name=name + '_data')
        
        # Initialize head and tail to zero if creating the segment
        if metadata_shm.buf[:4] == b'\x00' * 4:
            metadata_shm.buf[0:128] = b'\x00' * 128

        return LitSMQ(name, metadata_shm=metadata_shm, data_shm=data_shm)

    @staticmethod
    def attach(name):
        try:
            metadata_shm = shared_memory.SharedMemory(name=name + '_metadata')
            data_shm = shared_memory.SharedMemory(name=name + '_data')
            return LitSMQ(name, metadata_shm=metadata_shm, data_shm=data_shm)
        except FileNotFoundError as e:
            print(f"Error attaching shared memory: {e}")
            raise e

    def put(self, item):
        item_bytes = pickle.dumps(item)
        item_size = len(item_bytes)
        if item_size + 4 > self.data_size:
            raise ValueError("Item size exceeds queue capacity")

        with self.lock:
            head = int.from_bytes(self.metadata_buffer[self.head_index:self.head_index+4], 'little')
            tail = int.from_bytes(self.metadata_buffer[self.tail_index:self.tail_index+4], 'little')

            if tail + item_size + 4 > self.data_size:
                raise ValueError("Queue is full")

            self.data_buffer[tail:tail + 4] = item_size.to_bytes(4, 'little')
            tail += 4
            self.data_buffer[tail:tail + item_size] = item_bytes
            tail += item_size
            self.metadata_buffer[self.tail_index:self.tail_index+4] = tail.to_bytes(4, 'little')

    def get(self):
        with self.lock:
            head = int.from_bytes(self.metadata_buffer[self.head_index:self.head_index+4], 'little')
            tail = int.from_bytes(self.metadata_buffer[self.tail_index:self.tail_index+4], 'little')

            if head == tail:
                return None  # Queue is empty

            item_size = int.from_bytes(self.data_buffer[head:head + 4], 'little')
            head += 4
            item_bytes = self.data_buffer[head:head + item_size]
            head += item_size
            self.metadata_buffer[self.head_index:self.head_index+4] = head.to_bytes(4, 'little')

            item = pickle.loads(item_bytes)
            return item

    def close(self):
        self.metadata_shm.close()
        self.data_shm.close()

    def unlink(self):
        self.metadata_shm.unlink()
        self.data_shm.unlink()

    def get_shared_memory_names(self):
        return self.metadata_shm.name, self.data_shm.name

def cleanup_shared_memory(metadata_shm_name, data_shm_name):
    try:
        metadata_shm = shared_memory.SharedMemory(name=metadata_shm_name)
        metadata_shm.unlink()
        metadata_shm.close()
        print(f"Unlinked and closed shared memory: {metadata_shm_name}")
    except FileNotFoundError:
        print(f"Shared memory {metadata_shm_name} not found for cleanup.")

    try:
        data_shm = shared_memory.SharedMemory(name=data_shm_name)
        data_shm.unlink()
        data_shm.close()
        print(f"Unlinked and closed shared memory: {data_shm_name}")
    except FileNotFoundError:
        print(f"Shared memory {data_shm_name} not found for cleanup.")



# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")

# timeout when we need to poll or wait indefinitely for a result in a loop.
LONG_TIMEOUT = 100


def _inject_context(context: Union[List[dict], dict], func, *args, **kwargs):
    sig = inspect.signature(func)
    if "context" in sig.parameters:
        return func(*args, **kwargs, context=context)
    return func(*args, **kwargs)


@log_time
def get_batch_from_uid(uids, lit_api, request_buffer):
    batches = []
    for uid in uids:
        try:
            x_enc, pipe_s = request_buffer.pop(uid)
        except KeyError:
            continue
        batches.append((x_enc, pipe_s))
    return batches


# @log_time
def collate_requests(
    lit_api: LitAPI, request_queue: mp.Queue, request_buffer: Dict, max_batch_size: int, batch_timeout: float, sl, offset, sm_queue
) -> Optional[List]:
    curr = sl[0]
    uids = []
    entered_at = time.time()
    end_time = entered_at + batch_timeout

    while time.time() < end_time and len(uids) < max_batch_size:
        remaining_time = end_time - time.time()
        if remaining_time <= 0:
            break
        item = sm_queue.get()
        if item is not None:
            uids.append(item)

    if uids:
        return offset, uids

    return None

# @log_time
def consumer_collate(
    lit_api: LitAPI, request_queue: Queue, request_buffer: Dict, max_batch_size: int, batch_timeout: float
) -> Optional[List]:
    uids = []
    entered_at = time.time()
    end_time = entered_at + batch_timeout

    while time.time() < end_time and len(uids) < max_batch_size:
        remaining_time = end_time - time.time()
        if remaining_time <= 0:
            break

        try:
            uid = request_queue.get(timeout=min(remaining_time, 0.001))
            uids.append(uid)
        except Empty:
            continue

    if uids:
        return uids

    return None


def run_single_loop(lit_api: LitAPI, lit_spec: LitSpec, request_queue: mp.Queue, request_buffer: Dict):
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


def run_batched_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: mp.Queue,
    request_buffer: Dict,
    max_batch_size: int,
    batch_timeout: float,
    sl: List,
    result_sml: List,
    queue_name,
    metadata_shm_name,
    data_shm_name

):
    queue = LitSMQ.attach(queue_name)
    # shared_queue = SimpleSharedMemoryQueue.create(sm_queue)
    offset = 0
    while True:
        t0 = time.time()
        uids = collate_requests(
            lit_api,
            request_queue,
            request_buffer,
            max_batch_size,
            batch_timeout,
            sl,
            offset,
            queue
        )
        t1 = time.time()
        if not uids:
            continue
        offset, items = uids
        batches = [e[1] for e in items]
        uids = [e[0] for e in items]
        print(uids)
        server_logger.info(f"batch_wait_time (ms), {(t1 - t0) * 1000}")
        # batches = get_batch_from_uid(uids, lit_api, request_buffer)
        server_logger.info(f"batch_size, {len(batches)}")

        logger.debug(f"{len(batches)} batched requests received")
        pipes = batches
        inputs = [1]* len(pipes)  # fake
        # inputs, pipes = zip(*batches)

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
            with Timing("inference"):
                y = _inject_context(contexts, lit_api.predict, x)
            outputs = lit_api.unbatch(y)
            for y, pipe_s, context, uid in zip(outputs, pipes, contexts, uids):
                y_enc = _inject_context(context, lit_api.encode_response, y)

                with contextlib.suppress(BrokenPipeError):
                    # pipe_send(pipe_s, (y_enc, LitAPIStatus.OK))
                    result_sml[uid] = y_enc

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the batched request.\n"
                "Please check the error trace for more details."
            )
            err_pkl = pickle.dumps(e)
            with contextlib.suppress(BrokenPipeError):
                for pipe_s in pipes:
                    result_sml[uid] = -1


def run_streaming_loop(lit_api: LitAPI, lit_spec: LitSpec, request_queue: mp.Queue, request_buffer: Dict):
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


def run_batched_streaming_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: mp.Queue,
    request_buffer: Dict,
    max_batch_size: int,
    batch_timeout: float,
):
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
    device: str,
    worker_id: int,
    request_queue: Queue,
    request_buffer: Dict,
    max_batch_size: int,
    batch_timeout: float,
    stream: bool,
    sl: List,
    result_sml:List,
    queue_name, metadata_shm_name, data_shm_name
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
        run_batched_loop(lit_api, lit_spec, request_queue, request_buffer, max_batch_size, batch_timeout, sl, result_sml, queue_name, metadata_shm_name, data_shm_name)
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
        api_path: str = "/predict",
        stream: bool = False,
        spec: Optional[LitSpec] = None,
    ):
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


        self.queue = queue.Queue()

        self.api_path = api_path
        lit_api.stream = stream
        lit_api.sanitize(max_batch_size, spec=spec)
        self.app = FastAPI(lifespan=self.lifespan)
        # gzip does not play nicely with streaming, see https://github.com/tiangolo/fastapi/discussions/8448
        if not stream:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response

        self.lit_api = lit_api
        self.lit_spec = spec
        self.workers_per_device = workers_per_device
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.batch_timeout = batch_timeout
        self.stream = stream
        self._connector = _Connector(accelerator=accelerator, devices=devices)
        self.pipe_pool = [Pipe() for _ in range(1000)]
        self.queue = queue.Queue()

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
        queue_name = "simple_lit"
        self.sm_queue = LitSMQ.create(name=queue_name, data_size=10_100_000)
        metadata_shm_name, data_shm_name = self.sm_queue.get_shared_memory_names()

        def signal_handler(sig, frame):
            print("Signal received, cleaning up shared memory...")
            cleanup_shared_memory(metadata_shm_name, data_shm_name)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        smm = SharedMemoryManager()
        smm.start()
        items = [0] + [pickle.dumps([0, None])]*BUFFER_SIZE
        self.sl = smm.ShareableList(items)
        self.result_sml = smm.ShareableList([0]*BUFFER_SIZE)
                
        # manager = Manager()
        self.request_buffer = None #manager.dict()
        # ctx = mp.get_context("spawn")
        self.request_queue = None #ctx.Queue()

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
                    self.sl,
                    self.result_sml,
                    queue_name, metadata_shm_name, data_shm_name,
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


        t = threading.Thread(target=self.consumer, daemon=True)
        t.start()

        yield

        self.sm_queue.close()
        self.sm_queue.unlink()      
        smm.shutdown()
        for process, worker_id in process_list:
            logging.info(f"terminating worker worker_id={worker_id}")
            process.terminate()

    # @log_time
    def new_pipe(self) -> tuple:
        try:
            return self.pipe_pool.pop()
        except Exception:
            return Pipe()

    def close_pipe(self, pipe_s, pipe_r):
        self.pipe_pool.append((pipe_s, pipe_r))
        # pipe_s.close()
        # pipe_r.close()

    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    def get_from_pipe(self, read):
        while True:
            if read.poll(LONG_TIMEOUT):
                return read.recv()

    @log_time
    async def data_reader(self, uid, read):
        while True:
            if self.result_sml[uid]>0 or self.result_sml==-1:
                return self.result_sml[uid], ""
            await asyncio.sleep(0.001)
        # data_available = asyncio.Event()
        # loop = asyncio.get_running_loop()
        # loop.add_reader(read.fileno(), data_available.set)
        # if not read.poll():
        #     await data_available.wait()
        # loop.remove_reader(read.fileno())
        # # return read.recv()
        # return pipe_read(read)

    async def win_data_streamer(self, read, write, send_status=False):
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
                    yield response, status
                    return
                if send_status:
                    yield response, status
                else:
                    yield response

            await asyncio.sleep(0.0001)

    async def data_streamer(self, read: Connection, write: Connection, send_status=False):
        data_available = asyncio.Event()
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def reader():
            try:
                while read.poll():  # Check if there's data available to read
                    response, status = read.recv()
                    queue.put_nowait((response, status))
                    data_available.set()
            except Exception as e:
                logger.error(f"Exception in reader: {e}")

        loop.add_reader(read.fileno(), reader)

        try:
            while True:
                await data_available.wait()
                data_available.clear()

                while not queue.empty():
                    response, status = await queue.get()
                    if status == LitAPIStatus.FINISH_STREAMING:
                        loop.remove_reader(read.fileno())
                        return
                    if status == LitAPIStatus.ERROR:
                        logger.error(
                            "Error occurred while streaming outputs from the inference worker. "
                            "Please check the above traceback."
                        )
                        loop.remove_reader(read.fileno())
                        if send_status:
                            yield response, status
                        return
                    if send_status:
                        yield response, status
                    else:
                        yield response
        finally:
            loop.remove_reader(read.fileno())

    def cleanup_request(self, request_buffer, uid):
        with contextlib.suppress(KeyError):
            request_buffer.pop(uid)

    def consumer(self):
        print("Running consumer")
        offset = 1
        while True:
            offset = offset%BUFFER_SIZE
            t0 = time.perf_counter()
            items = consumer_collate(None, self.queue, {}, self.max_batch_size, self.batch_timeout)
            t1 = time.perf_counter()
            if items:
                server_logger.info(f"aggregate_requests (ms), {(t1 - t0) * 1000}")
                with Timing("put_request"):
                    # uid, buffer_data
                    # data = (uid, write)
                    for uid, data in items:
                        self.sm_queue.put((uid, data[1]))
                        item = pickle.dumps((uid, data[1]))
                        self.sl[offset] = item
                        self.sl[0] = offset
                        offset += 1

    def setup_server(self):
        self.uid = 0

        def get_uid():
            self.uid+=1
            return self.uid

        @self.app.get("/", dependencies=[Depends(self.setup_auth())])
        async def index(request: Request) -> Response:
            return Response(content="litserve running")

        @log_time
        async def predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
            uid = get_uid()
            logger.debug(f"Received request uid={uid}")

            # read, write = self.new_pipe()
            read, write = None, None

            if self.request_type == Request:
                if request.headers["Content-Type"] == "application/x-www-form-urlencoded" or request.headers[
                    "Content-Type"
                ].startswith("multipart/form-data"):
                    buffer_data = (await request.form(), write)
                else:
                    buffer_data = (await request.json(), write)
            else:
                buffer_data = (request, write)

            # with Timing("put request"):
            self.queue.put_nowait((uid, buffer_data))

            background_tasks.add_task(self.cleanup_request, self.request_buffer, uid)

            if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
                data = await wait_for_queue_timeout(
                    asyncio.to_thread(self.get_from_pipe, read), self.timeout, uid, self.request_buffer
                )
            else:
                data = await wait_for_queue_timeout(self.data_reader(uid, read), self.timeout, uid, self.request_buffer)

            # self.close_pipe(read, write)
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

        port_msg = f"port must be a value from 1024 to 65535 but got {port}"
        try:
            port = int(port)
        except ValueError:
            raise ValueError(port_msg)

        if not (1024 <= port <= 65535):
            raise ValueError(port_msg)

        uvicorn.run(host="0.0.0.0", port=port, app=self.app, log_level=log_level, **kwargs)

    def setup_auth(self):
        if hasattr(self.lit_api, "authorize") and callable(self.lit_api.authorize):
            return self.lit_api.authorize
        if LIT_SERVER_API_KEY:
            return api_key_auth
        return no_auth
