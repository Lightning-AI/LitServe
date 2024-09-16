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


def run_single_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    preprocess_queue: Optional[Queue],
    response_queues: List[Queue],
):
    while True:
        if preprocess_queue is None:
            # We are handling the full pipeline
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
                response_queues[response_queue_id].put((
                    uid,
                    (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR),
                ))
                continue

            try:
                context = {}
                if hasattr(lit_spec, "populate_context"):
                    lit_spec.populate_context(context, x_enc)
                x = _inject_context(context, lit_api.decode_request, x_enc)

                y = _inject_context(context, lit_api.predict, x)
                y_enc = _inject_context(context, lit_api.encode_response, y)
                response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))
            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the request uid=%s.\n"
                    "Please check the error trace for more details.",
                    uid,
                )
                err_pkl = pickle.dumps(e)
                response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))
        else:
            # We are in the inference worker, get data from preprocess_queue
            try:
                response_queue_id, uid, y = preprocess_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                context = {}
                if hasattr(lit_spec, "populate_context"):
                    lit_spec.populate_context(context, y)

                z = _inject_context(context, lit_api.predict, y)
                z_enc = _inject_context(context, lit_api.encode_response, z)
                response_queues[response_queue_id].put((uid, (z_enc, LitAPIStatus.OK)))
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
    preprocess_queue: Optional[Queue],
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
):
    while True:
        if preprocess_queue is None:
            # We are handling the full pipeline
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
                response_queues[response_queue_id].put((
                    uid,
                    (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR),
                ))

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
                x_batched = lit_api.batch(x)

                y_batched = _inject_context(contexts, lit_api.predict, x_batched)
                y_unbatched = lit_api.unbatch(y_batched)
                for response_queue_id, y, uid, context in zip(response_queue_ids, y_unbatched, uids, contexts):
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
        else:
            # We are in the inference worker, get data from preprocess_queue
            ready_to_inference_list = []

            # Collect items from the queue up to max_batch_size or until timeout
            try:
                item = preprocess_queue.get(timeout=batch_timeout)
                ready_to_inference_list.append(item)
            except Empty:
                pass

            if not ready_to_inference_list:
                continue

            # Unpack the collected items
            response_queue_ids, uids, inputs = ready_to_inference_list[0]

            try:
                contexts = [{}] * len(inputs)
                if hasattr(lit_spec, "populate_context"):
                    for input, context in zip(inputs, contexts):
                        lit_spec.populate_context(context, input)

                z_batched = _inject_context(contexts, lit_api.predict, inputs)
                z_unbatched = lit_api.unbatch(z_batched)

                for response_queue_id, z, uid, context in zip(response_queue_ids, z_unbatched, uids, contexts):
                    z_enc = _inject_context(context, lit_api.encode_response, z)
                    response_queues[response_queue_id].put((uid, (z_enc, LitAPIStatus.OK)))

            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the batched request.\n"
                    "Please check the error trace for more details."
                )
                err_pkl = pickle.dumps(e)
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


def run_streaming_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    preprocess_queue: Optional[Queue],
    response_queues: List[Queue],
):
    while True:
        if preprocess_queue is None:
            # We are handling the full pipeline
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
                response_queues[response_queue_id].put((
                    uid,
                    (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR),
                ))
                continue

            try:
                context = {}
                if hasattr(lit_spec, "populate_context"):
                    lit_spec.populate_context(context, x_enc)
                x = _inject_context(context, lit_api.decode_request, x_enc)

                y_gen = _inject_context(context, lit_api.predict, x)
                y_enc_gen = _inject_context(context, lit_api.encode_response, y_gen)
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
                err_pkl = pickle.dumps(e)
                response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))
        else:
            # We are in the inference worker, get data from preprocess_queue
            try:
                response_queue_id, uid, y = preprocess_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                context = {}
                if hasattr(lit_spec, "populate_context"):
                    lit_spec.populate_context(context, y)

                z_gen = _inject_context(context, lit_api.predict, y)
                z_enc_gen = _inject_context(context, lit_api.encode_response, z_gen)
                for z_enc in z_enc_gen:
                    z_enc = lit_api.format_encoded_response(z_enc)
                    response_queues[response_queue_id].put((uid, (z_enc, LitAPIStatus.OK)))
                response_queues[response_queue_id].put((uid, ("", LitAPIStatus.FINISH_STREAMING)))
            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                    "Please check the error trace for more details.",
                    uid,
                )
                err_pkl = pickle.dumps(e)
                response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


def run_batched_streaming_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    preprocess_queue: Optional[Queue],
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
):
    while True:
        if preprocess_queue is None:
            # We are handling the full pipeline
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
                response_queues[response_queue_id].put((
                    uid,
                    (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR),
                ))

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
                x_batched = lit_api.batch(x)

                y_iter = _inject_context(contexts, lit_api.predict, x_batched)
                unbatched_iter = lit_api.unbatch(y_iter)
                y_enc_iter = _inject_context(contexts, lit_api.encode_response, unbatched_iter)

                for y_batch in y_enc_iter:
                    for response_queue_id, y_enc, uid in zip(response_queue_ids, y_batch, uids):
                        y_enc = lit_api.format_encoded_response(y_enc)
                        response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    response_queues[response_queue_id].put((uid, ("", LitAPIStatus.FINISH_STREAMING)))
            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the batched streaming request.\n"
                    "Please check the error trace for more details."
                )
                err_pkl = pickle.dumps(e)
                response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))
        else:
            # We are in the inference worker, get data from preprocess_queue
            ready_to_inference_list = []

            # Collect items from the queue up to max_batch_size or until timeout
            try:
                item = preprocess_queue.get(timeout=batch_timeout)
                ready_to_inference_list.append(item)
            except Empty:
                pass

            if not ready_to_inference_list:
                continue

            # Unpack the collected items
            response_queue_ids, uids, inputs = ready_to_inference_list[0]

            try:
                contexts = [{}] * len(inputs)
                if hasattr(lit_spec, "populate_context"):
                    for input, context in zip(inputs, contexts):
                        lit_spec.populate_context(context, input)

                y_iter = _inject_context(contexts, lit_api.predict, inputs)
                unbatched_iter = lit_api.unbatch(y_iter)
                y_enc_iter = _inject_context(contexts, lit_api.encode_response, unbatched_iter)

                for y_batch in y_enc_iter:
                    for response_queue_id, y_enc, uid in zip(response_queue_ids, y_batch, uids):
                        y_enc = lit_api.format_encoded_response(y_enc)
                        response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    response_queues[response_queue_id].put((uid, ("", LitAPIStatus.FINISH_STREAMING)))
            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the batched streaming request.\n"
                    "Please check the error trace for more details."
                )
                err_pkl = pickle.dumps(e)
                response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


def inference_worker(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    device: str,
    worker_id: int,
    request_queue: Queue,
    preprocess_queue: Optional[Queue],
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
    stream: bool,
    workers_setup_status: Dict[str, bool] = None,
):
    lit_api.setup(device)
    lit_api.device = device

    print(f"Setup complete for inference worker {worker_id}.")

    if workers_setup_status:
        workers_setup_status[worker_id] = True

    if stream:
        if max_batch_size > 1:
            run_batched_streaming_loop(
                lit_api,
                lit_spec,
                request_queue,
                preprocess_queue,
                response_queues,
                max_batch_size,
                batch_timeout,
            )
        else:
            run_streaming_loop(
                lit_api,
                lit_spec,
                request_queue,
                preprocess_queue,
                response_queues,
            )
    else:
        if max_batch_size > 1:
            run_batched_loop(
                lit_api,
                lit_spec,
                request_queue,
                preprocess_queue,
                response_queues,
                max_batch_size,
                batch_timeout,
            )
        else:
            run_single_loop(
                lit_api,
                lit_spec,
                request_queue,
                preprocess_queue,
                response_queues,
            )


def run_single_preprocess_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    preprocess_queue: Queue,
    response_queues: List[Queue],
):
    while True:
        try:
            response_queue_id, uid, timestamp, x_enc = request_queue.get(timeout=1.0)
        except Empty:
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
                lit_api.preprocess,
                x,
            )
            preprocess_queue.put((response_queue_id, uid, y))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while preprocessing the request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            err_pkl = pickle.dumps(e)
            response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


def run_batched_preprocess_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    preprocess_queue: Queue,
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
            x_batched = lit_api.batch(x)
            y_batched = _inject_context(contexts, lit_api.preprocess, x_batched)

            preprocess_queue.put([response_queue_ids, uids, y_batched])

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while preprocessing the batched request.\n"
                "Please check the error trace for more details."
            )
            err_pkl = pickle.dumps(e)
            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, (err_pkl, LitAPIStatus.ERROR)))


def preprocess_worker(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    device: str,
    worker_id: int,
    request_queue: Queue,
    preprocess_queue: Queue,
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
):
    print(f"Preprocess worker {worker_id}.")

    if max_batch_size > 1:
        run_batched_preprocess_loop(
            lit_api, lit_spec, request_queue, preprocess_queue, response_queues, max_batch_size, batch_timeout
        )
    else:
        run_single_preprocess_loop(lit_api, lit_spec, request_queue, preprocess_queue, response_queues)
