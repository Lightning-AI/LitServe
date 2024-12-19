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
import sys
import time
from abc import ABC
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException
from starlette.formparsers import MultiPartParser

from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, PickleableHTTPException, WorkerSetupStatus

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

        except HTTPException as e:
            response_queues[response_queue_id].put((
                uid,
                (PickleableHTTPException.from_exception(e), LitAPIStatus.ERROR),
            ))

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            response_queues[response_queue_id].put((uid, (e, LitAPIStatus.ERROR)))


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

        except HTTPException as e:
            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((
                    uid,
                    (PickleableHTTPException.from_exception(e), LitAPIStatus.ERROR),
                ))

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the batched request.\n"
                "Please check the error trace for more details."
            )
            for response_queue_id, uid in zip(response_queue_ids, uids):
                response_queues[response_queue_id].put((uid, (e, LitAPIStatus.ERROR)))


def run_streaming_loop(
    lit_api: LitAPI,
    lit_spec: LitSpec,
    request_queue: Queue,
    response_queues: List[Queue],
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
            for y_enc in y_enc_gen:
                y_enc = lit_api.format_encoded_response(y_enc)
                response_queues[response_queue_id].put((uid, (y_enc, LitAPIStatus.OK)))
            response_queues[response_queue_id].put((uid, ("", LitAPIStatus.FINISH_STREAMING)))

        except HTTPException as e:
            response_queues[response_queue_id].put((
                uid,
                (PickleableHTTPException.from_exception(e), LitAPIStatus.ERROR),
            ))
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            response_queues[response_queue_id].put((uid, (e, LitAPIStatus.ERROR)))


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

        except HTTPException as e:
            response_queues[response_queue_id].put((
                uid,
                (PickleableHTTPException.from_exception(e), LitAPIStatus.ERROR),
            ))

        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming batched request.\n"
                "Please check the error trace for more details."
            )
            response_queues[response_queue_id].put((uid, (e, LitAPIStatus.ERROR)))


class _BaseLoop(ABC):
    """Loop runs an inference engine that executes a specific set of hooks, implemented in the LitAPI, in a predefined
    order.

    For a default loop, LitAPI must implement the following hooks:
      - decode_request
      - batch
      - predict
      - unbatch
      - encode_response

    To implement a custom loop, subclass this class and implement the `run` method. The `run` method should execute the
    hooks in the desired order.

    `__call__` method is the entry point for the worker process. It calls the `run` method in a loop until the worker is
    terminated.

    Example:

    ```python
    class TestLoop(_BaseLoop):
        def run(
            self,
            lit_api: LitAPI,
            lit_spec: Optional[LitSpec],
            device: str,
            worker_id: int,
            request_queue: Queue,
            response_queues: List[Queue],
            max_batch_size: int,
            batch_timeout: float,
            stream: bool,
            workers_setup_status: Dict[int, str],
            callback_runner: CallbackRunner,
        ):
            item = request_queue.get()
            if item is None:
                return

            response_queue_id, uid, timestamp, x_enc = item
            # Expects LitAPI to implement the load_cache method
            lit_api.load_cache(x_enc)
            x = lit_api.decode_request(x_enc)
            response = lit_api.predict(x)
            response_enc = lit_api.encode_response(response)
            response_queues[response_queue_id].put((uid, (response_enc, LitAPIStatus.OK)))
    ```

    """

    def pre_setup(self, lit_api: LitAPI, spec: Optional[LitSpec]):
        pass

    async def run_in_background(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        request_queue: Queue,
        max_batch_size: int,
        batch_timeout: float,
        response_queues: List[Queue],
    ):
        pass

    def __call__(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        response_queues: List[Queue],
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        if asyncio.iscoroutinefunction(self.run):
            loop = asyncio.new_event_loop()

            async def _wrapper():
                print("Running LitLoop in a asyncio event loop")
                future = self.run_in_background(
                    lit_api, lit_spec, request_queue, max_batch_size, batch_timeout, response_queues
                )
                loop.create_task(future)
                while True:
                    try:
                        await self.run(
                            lit_api,
                            lit_spec,
                            device,
                            worker_id,
                            request_queue,
                            response_queues,
                            max_batch_size,
                            batch_timeout,
                            stream,
                            workers_setup_status,
                            callback_runner,
                        )
                        await asyncio.sleep(0.00001)
                    except Exception as e:
                        logger.exception("An error occurred in the loop: %s", e)
                        # Optionally, break the loop or handle the error as needed

            loop.run_until_complete(_wrapper())
        else:
            while True:
                self.run(
                    lit_api,
                    lit_spec,
                    device,
                    worker_id,
                    request_queue,
                    response_queues,
                    max_batch_size,
                    batch_timeout,
                    stream,
                    workers_setup_status,
                    callback_runner,
                )

    def run(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        response_queues: List[Queue],
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        raise NotImplementedError


class LitLoop(_BaseLoop):
    def __init__(self):
        self._context = {}

    def get_batch_requests(self, lit_api: LitAPI, request_queue: Queue, max_batch_size: int, batch_timeout: float):
        batches, timed_out_uids = collate_requests(
            lit_api,
            request_queue,
            max_batch_size,
            batch_timeout,
        )
        return batches, timed_out_uids

    def get_request(self, request_queue: Queue, block: bool = True, timeout: Optional[float] = None):
        try:
            return request_queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def populate_context(self, lit_spec: LitSpec, request: Any):
        if lit_spec and hasattr(lit_spec, "populate_context"):
            lit_spec.populate_context(self._context, request)

    def put_response(
        self, response_queues: List[Queue], response_queue_id: int, uid: str, response_data: Any, status: LitAPIStatus
    ) -> None:
        response_queues[response_queue_id].put((uid, (response_data, status)))

    def put_error_response(
        self, response_queues: List[Queue], response_queue_id: int, uid: str, error: Exception
    ) -> None:
        response_queues[response_queue_id].put((uid, (error, LitAPIStatus.ERROR)))


class DefaultLoop(LitLoop):
    def pre_setup(self, lit_api: LitAPI, spec: Optional[LitSpec]):
        # we will sanitize regularly if no spec
        # in case, we have spec then:
        # case 1: spec implements a streaming API
        # Case 2: spec implements a non-streaming API
        if spec:
            # TODO: Implement sanitization
            lit_api._spec = spec
            return

        original = lit_api.unbatch.__code__ is LitAPI.unbatch.__code__
        if (
            lit_api.stream
            and lit_api.max_batch_size > 1
            and not all([
                inspect.isgeneratorfunction(lit_api.predict),
                inspect.isgeneratorfunction(lit_api.encode_response),
                (original or inspect.isgeneratorfunction(lit_api.unbatch)),
            ])
        ):
            raise ValueError(
                """When `stream=True` with max_batch_size > 1, `lit_api.predict`, `lit_api.encode_response` and
                `lit_api.unbatch` must generate values using `yield`.

             Example:

                def predict(self, inputs):
                    ...
                    for i in range(max_token_length):
                        yield prediction

                def encode_response(self, outputs):
                    for output in outputs:
                        encoded_output = ...
                        yield encoded_output

                def unbatch(self, outputs):
                    for output in outputs:
                        unbatched_output = ...
                        yield unbatched_output
             """
            )

        if lit_api.stream and not all([
            inspect.isgeneratorfunction(lit_api.predict),
            inspect.isgeneratorfunction(lit_api.encode_response),
        ]):
            raise ValueError(
                """When `stream=True` both `lit_api.predict` and
             `lit_api.encode_response` must generate values using `yield`.

             Example:

                def predict(self, inputs):
                    ...
                    for i in range(max_token_length):
                        yield prediction

                def encode_response(self, outputs):
                    for output in outputs:
                        encoded_output = ...
                        yield encoded_output
             """
            )


class SingleLoop(DefaultLoop):
    def __call__(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        response_queues: List[Queue],
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        run_single_loop(lit_api, lit_spec, request_queue, response_queues, callback_runner)


class BatchedLoop(DefaultLoop):
    def __call__(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        response_queues: List[Queue],
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        run_batched_loop(
            lit_api,
            lit_spec,
            request_queue,
            response_queues,
            max_batch_size,
            batch_timeout,
            callback_runner,
        )


class StreamingLoop(DefaultLoop):
    def __call__(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        response_queues: List[Queue],
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        run_streaming_loop(lit_api, lit_spec, request_queue, response_queues, callback_runner)


class BatchedStreamingLoop(DefaultLoop):
    def __call__(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        response_queues: List[Queue],
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        run_batched_streaming_loop(
            lit_api,
            lit_spec,
            request_queue,
            response_queues,
            max_batch_size,
            batch_timeout,
            callback_runner,
        )


def get_default_loop(stream: bool, max_batch_size: int) -> _BaseLoop:
    return (
        BatchedStreamingLoop()
        if stream and max_batch_size > 1
        else StreamingLoop()
        if stream
        else BatchedLoop()
        if max_batch_size > 1
        else SingleLoop()
    )


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
    workers_setup_status: Dict[int, str],
    callback_runner: CallbackRunner,
    loop: Union[str, _BaseLoop],
):
    callback_runner.trigger_event(EventTypes.BEFORE_SETUP, lit_api=lit_api)
    try:
        lit_api.setup(device)
    except Exception:
        logger.exception(f"Error setting up worker {worker_id}.")
        workers_setup_status[worker_id] = WorkerSetupStatus.ERROR
        return
    lit_api.device = device
    callback_runner.trigger_event(EventTypes.AFTER_SETUP, lit_api=lit_api)

    print(f"Setup complete for worker {worker_id}.")

    if workers_setup_status:
        workers_setup_status[worker_id] = WorkerSetupStatus.READY

    if lit_spec:
        logging.info(f"LitServe will use {lit_spec.__class__.__name__} spec")

    if loop == "auto":
        loop = get_default_loop(stream, max_batch_size)

    loop(
        lit_api,
        lit_spec,
        device,
        worker_id,
        request_queue,
        response_queues,
        max_batch_size,
        batch_timeout,
        stream,
        workers_setup_status,
        callback_runner,
    )
