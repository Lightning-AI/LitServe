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
import inspect
import logging
import multiprocessing as mp
import pickle
import sys
import time
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple, Union
from fastapi import HTTPException
from starlette.formparsers import MultiPartParser

from litserve import LitAPI
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus


logging.basicConfig(filename="server.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


mp.allow_connection_pickling()

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

except ImportError:
    print(
        "uvloop is not installed. Falling back to the default asyncio event loop. "
        "Please install uvloop for better performance using `pip install uvloop`."
    )

logger = logging.getLogger(__name__)

# FastAPI writes form files to disk over 1MB by default, which prevents serialization by multiprocessing
MultiPartParser.max_file_size = sys.maxsize


def _inject_context(context: Union[List[dict], dict], func, *args, **kwargs):
    sig = inspect.signature(func)
    if "context" in sig.parameters:
        return func(*args, **kwargs, context=context)
    return func(*args, **kwargs)


def collate_requests(
    lit_api: LitAPI, request_queue: Queue, max_batch_size: int, batch_timeout: float
) -> Tuple[List, List]:
    payloads = []
    timed_out_uids = []
    entered_at = time.monotonic()
    end_time = entered_at + batch_timeout
    apply_timeout = lit_api.request_timeout not in (-1, False)

    if batch_timeout == 0:
        while len(payloads) < max_batch_size:
            try:
                response_queue_id, uid, timestamp, x_enc = request_queue.get_nowait()
                if apply_timeout and time.monotonic() - timestamp > lit_api.request_timeout:
                    timed_out_uids.append((response_queue_id, uid))
                else:
                    payloads.append((response_queue_id, uid, x_enc))
            except Empty:
                break
        return payloads, timed_out_uids

    while time.monotonic() < end_time and len(payloads) < max_batch_size:
        remaining_time = end_time - time.monotonic()
        if remaining_time <= 0:
            break

        try:
            response_queue_id, uid, timestamp, x_enc = request_queue.get(timeout=min(remaining_time, 0.001))
            if apply_timeout and time.monotonic() - timestamp > lit_api.request_timeout:
                timed_out_uids.append((response_queue_id, uid))
            else:
                payloads.append((response_queue_id, uid, x_enc))

        except Empty:
            continue

    return payloads, timed_out_uids


async def run_heter_pipeline(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
    heter_pipeline: Queue,
):
    cpu_batch = []
    gpu_batch = []
    cpu_to_gpu_queue = Queue()
    last_cpu_process_time = time.time()
    last_gpu_process_time = time.time()
    processing_gpu = False

    loop = asyncio.get_event_loop()

    async def process_cpu_batch():
        nonlocal cpu_batch, last_cpu_process_time
        while True:
            if cpu_batch:
                print(f"Processing batch of {len(cpu_batch)} requests on CPU")
                cpu_results = await loop.run_in_executor(None, lit_api.process_on_cpu, cpu_batch)
                for result, (response_queue_id, uid, _, _) in zip(cpu_results, cpu_batch):
                    cpu_to_gpu_queue.put((response_queue_id, uid, result))
                print(f"Completed CPU processing for batch of {len(cpu_batch)} requests")
                cpu_batch = []
                last_cpu_process_time = time.time()
            await asyncio.sleep(0.001)

    async def process_gpu_batch():
        nonlocal gpu_batch, last_gpu_process_time, processing_gpu
        while True:
            if gpu_batch:
                print(f"Processing batch of {len(gpu_batch)} requests on GPU")
                gpu_results = await loop.run_in_executor(None, lit_api.process_on_gpu, gpu_batch)
                for (response_queue_id, uid, _), result in zip(gpu_batch, gpu_results):
                    y_enc = lit_api.encode_response(result)
                    response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))
                print(f"Completed GPU processing for batch of {len(gpu_batch)} requests")
                gpu_batch = []
                last_gpu_process_time = time.time()
                processing_gpu = False
            await asyncio.sleep(0.001)

    async def fill_cpu_batch():
        nonlocal cpu_batch
        while True:
            while len(cpu_batch) < max_batch_size:
                try:
                    response_queue_id, uid, timestamp, x_enc = request_queue.get_nowait()
                    cpu_batch.append((response_queue_id, uid, timestamp, x_enc))
                    print(f"Added request uid={uid} to CPU batch")
                except Empty:
                    break
            await asyncio.sleep(0.001)

    async def move_cpu_results_to_gpu_batch():
        nonlocal gpu_batch
        while True:
            while not cpu_to_gpu_queue.empty():
                gpu_batch.append(cpu_to_gpu_queue.get_nowait())
                print(f"Moved request to GPU batch, current GPU batch size: {len(gpu_batch)}")
            await asyncio.sleep(0.001)

    # Start CPU and GPU processing concurrently
    cpu_task = asyncio.create_task(process_cpu_batch())
    gpu_task = asyncio.create_task(process_gpu_batch())
    fill_cpu_task = asyncio.create_task(fill_cpu_batch())
    move_to_gpu_task = asyncio.create_task(move_cpu_results_to_gpu_batch())

    # Wait for all tasks to complete
    await asyncio.gather(cpu_task, gpu_task, fill_cpu_task, move_to_gpu_task)

    # Ensure to break the loop when all tasks are done
    while not request_queue.empty() or cpu_batch or gpu_batch or not cpu_to_gpu_queue.empty():
        await asyncio.sleep(0.001)

    print("All batches processed, exiting...")


def run_single_loop(lit_api: LitAPI, lit_spec: LitSpec, request_queue: Queue, response_queues: List[Queue]):
    while True:
        try:
            response_queue_id, uid, timestamp, x_enc = request_queue.get(timeout=1.0)
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
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))
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
            response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            err_pkl = pickle.dumps(e)
            response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


def run_batched_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    response_queues: List[Queue],
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

        for response_queue_id, uid in timed_out_uids:
            logger.error(
                f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                "has been timed out. "
                "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
            )
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))

        if not batches:
            continue
        logger.debug(f"{len(batches)} batched requests received")
        response_queue_ids, uids, inputs = zip(*batches)
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
            for response_queue_id, y, uid, context in zip(response_queue_ids, outputs, uids, contexts):
                y_enc = _inject_context(context, lit_api.encode_response, y)

                response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the batched request.\n"
                "Please check the error trace for more details."
            )
            err_pkl = pickle.dumps(e)
            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


def run_streaming_loop(lit_api: LitAPI, lit_spec: LitSpec, request_queue: Queue, response_queues: List[Queue]):
    while True:
        try:
            response_queue_id, uid, timestamp, x_enc = request_queue.get(timeout=1.0)
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
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))
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
                response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))
            response_queues[response_queue_id].put((uid, ("", LitAPIStatus.FINISH_STREAMING)))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            response_queues[response_queue_id].put((uid, (pickle.dumps(e), LitAPIStatus.ERROR)))


def run_batched_streaming_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    response_queues: List[Queue],
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
        for response_queue_id, uid in timed_out_uids:
            logger.error(
                f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                "has been timed out. "
                "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
            )
            response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))

        if not batches:
            continue
        response_queue_ids, uids, inputs = zip(*batches)
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
                for response_queue_id, y_enc, uid in zip(response_queue_ids, y_batch, uids):
                    y_enc = lit_api.format_encoded_response(y_enc)
                    response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))

            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, ("", LitAPIStatus.FINISH_STREAMING)))

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming batched request.\n"
                "Please check the error trace for more details."
            )
            err_pkl = pickle.dumps(e)
            response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


async def inference_worker(
    lit_api: LitAPI,
    lit_spec: Optional[LitSpec],
    device: str,
    worker_id: int,
    request_queue: Queue,
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
    stream: bool,
    workers_setup_status: Dict[str, bool] = None,
    heter_pipeline: Queue = None,
):
    print(f"Starting inference_worker {worker_id} on device {device}")

    lit_api.setup(device)
    lit_api.device = device

    print(f"Setup complete for worker {worker_id}.")

    if workers_setup_status:
        workers_setup_status[worker_id] = True

    if lit_spec:
        print(f"LitServe will use {lit_spec.__class__.__name__} spec")

    if heter_pipeline is not None:
        print(f"Worker {worker_id} using heter pipeline")
        await run_heter_pipeline(
            lit_api, lit_spec, request_queue, response_queues, max_batch_size, batch_timeout, heter_pipeline
        )
    elif stream:
        print(f"Worker {worker_id} using streaming")
        if max_batch_size > 1:
            run_batched_streaming_loop(lit_api, lit_spec, request_queue, response_queues, max_batch_size, batch_timeout)
        else:
            run_streaming_loop(lit_api, lit_spec, request_queue, response_queues)
    else:
        print(f"Worker {worker_id} using non-streaming")
        if max_batch_size > 1:
            run_batched_loop(lit_api, lit_spec, request_queue, response_queues, max_batch_size, batch_timeout)
        else:
            run_single_loop(lit_api, lit_spec, request_queue, response_queues)


def run_inference_worker(*args, **kwargs):
    asyncio.run(inference_worker(*args, **kwargs))
