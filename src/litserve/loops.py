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
from litserve.callbacks import CallbackRunner, EventTypes
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
    response_queues: List[Queue],
    callback_runner: CallbackRunner,
):
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

            callback_runner.trigger_event(EventTypes.BEFORE_DECODE_REQUEST, lit_api=lit_api)
            x = _inject_context(
                context,
                lit_api.decode_request,
                x_enc,
            )
            callback_runner.trigger_event(EventTypes.AFTER_DECODE_REQUEST, lit_api=lit_api)

            callback_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=lit_api)
            y = _inject_context(
                context,
                lit_api.predict,
                x,
            )
            callback_runner.trigger_event(EventTypes.AFTER_PREDICT, lit_api=lit_api)

            callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE, lit_api=lit_api)
            y_enc = _inject_context(
                context,
                lit_api.encode_response,
                y,
            )
            callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE, lit_api=lit_api)

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
    callback_runner: CallbackRunner,
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

            callback_runner.trigger_event(EventTypes.BEFORE_DECODE_REQUEST, lit_api=lit_api)
            x = [
                _inject_context(
                    context,
                    lit_api.decode_request,
                    input,
                )
                for input, context in zip(inputs, contexts)
            ]
            callback_runner.trigger_event(EventTypes.AFTER_DECODE_REQUEST, lit_api=lit_api)

            x = lit_api.batch(x)

            callback_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=lit_api)
            y = _inject_context(contexts, lit_api.predict, x)
            callback_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=lit_api)

            outputs = lit_api.unbatch(y)

            callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE, lit_api=lit_api)
            y_enc_list = []
            for response_queue_id, y, uid, context in zip(response_queue_ids, outputs, uids, contexts):
                y_enc = _inject_context(context, lit_api.encode_response, y)
                y_enc_list.append((response_queue_id, uid, y_enc))
            callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE, lit_api=lit_api)

            for response_queue_id, uid, y_enc in y_enc_list:
                response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))

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
    response_queues: List[Queue],
    request_evicted_status: Dict[str, bool],
    callback_runner: CallbackRunner,
):
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

            callback_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=lit_api)
            y_gen = _inject_context(
                context,
                lit_api.predict,
                x,
            )
            callback_runner.trigger_event(EventTypes.AFTER_PREDICT, lit_api=lit_api)

            y_enc_gen = _inject_context(
                context,
                lit_api.encode_response,
                y_gen,
            )
            check_interval = 50
            for index, y_enc in enumerate(y_enc_gen):
                if index % check_interval == 0 and request_evicted_status.get(uid):
                    request_evicted_status.pop(uid)
                    break
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
    callback_runner: CallbackRunner,
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

            callback_runner.trigger_event(EventTypes.BEFORE_DECODE_REQUEST, lit_api=lit_api)
            x = [
                _inject_context(
                    context,
                    lit_api.decode_request,
                    input,
                )
                for input, context in zip(inputs, contexts)
            ]
            callback_runner.trigger_event(EventTypes.AFTER_DECODE_REQUEST, lit_api=lit_api)

            x = lit_api.batch(x)

            callback_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=lit_api)
            y_iter = _inject_context(contexts, lit_api.predict, x)
            callback_runner.trigger_event(EventTypes.AFTER_PREDICT, lit_api=lit_api)

            unbatched_iter = lit_api.unbatch(y_iter)

            callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE, lit_api=lit_api)
            y_enc_iter = _inject_context(contexts, lit_api.encode_response, unbatched_iter)
            callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE, lit_api=lit_api)

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


def inference_worker(
    lit_api: LitAPI,
    lit_spec: Optional[LitSpec],
    device: str,
    worker_id: int,
    request_queue: Queue,
    response_queues: List[Queue],
    max_batch_size: int,
    batch_timeout: float,
    stream: bool,
    workers_setup_status: Dict[str, bool],
    request_evicted_status: Dict[str, bool],
    callback_runner: CallbackRunner,
):
    callback_runner.trigger_event(EventTypes.BEFORE_SETUP, lit_api=lit_api)
    lit_api.setup(device)
    lit_api.device = device
    callback_runner.trigger_event(EventTypes.AFTER_SETUP, lit_api=lit_api)

    print(f"Setup complete for worker {worker_id}.")

    if workers_setup_status:
        workers_setup_status[worker_id] = True

    if lit_spec:
        logging.info(f"LitServe will use {lit_spec.__class__.__name__} spec")
    if stream:
        if max_batch_size > 1:
            run_batched_streaming_loop(
                lit_api, lit_spec, request_queue, response_queues, max_batch_size, batch_timeout, callback_runner
            )
        else:
            run_streaming_loop(
                lit_api, lit_spec, request_queue, response_queues, request_evicted_status, callback_runner
            )
        return

    if max_batch_size > 1:
        run_batched_loop(
            lit_api, lit_spec, request_queue, response_queues, max_batch_size, batch_timeout, callback_runner
        )
    else:
        run_single_loop(lit_api, lit_spec, request_queue, response_queues, callback_runner)
