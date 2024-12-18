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
import sys
import time
from abc import ABC
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException
from starlette.formparsers import MultiPartParser

from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, PickleableHTTPException, WorkerSetupStatus

mp.allow_connection_pickling()

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

except ImportError:
    if sys.platform != "win32":
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

    def get_request(self, request_queue: Queue, timeout: float = 1.0):
        try:
            return request_queue.get(timeout=timeout)
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


def notify_timed_out_requests(
    response_queues: List[Queue],
    timed_out_uids: List[Tuple[int, str]],
):
    for response_queue_id, uid in timed_out_uids:
        logger.error(f"Request {uid} was waiting in the queue for too long and has been timed out.")
        response_queues[response_queue_id].put((uid, (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR)))


@dataclass
class Output:
    """Outputs from a single step of the loop."""

    uid: str
    output: Any
    status: LitAPIStatus


class ContinuousBatchingLoop(LitLoop):
    def __init__(self, max_sequence_length: int = 2048):
        """Runs continuous batching loop. This loop handles adding new requests, processing them in batches, and
        managing the state of active sequences.

        The loop requires the following methods to be implemented in the LitAPI:
          - setup: sets up the model on the device
          - decode_request: decodes the client request into a format that can be processed by the model
          - step: generates a new token for each sequence
          - encode_response: encodes the response into a format that can be sent to the client
          - has_finished: checks if the sequence has finished generating

        Args:
            max_sequence_length (int): The maximum sequence length allowed for any active sequence.

        """
        super().__init__()
        self.active_sequences: Dict[str, Dict] = {}  # uid -> {input, current_length, generated_sequence}
        self.max_sequence_length = max_sequence_length
        self.response_queue_ids: Dict[str, int] = {}  # uid -> response_queue_id

    def pre_setup(self, lit_api: LitAPI, spec: Optional[LitSpec]):
        if not lit_api.stream:
            raise ValueError(
                "Continuous batching loop requires streaming to be enabled. Please set LitServe(..., stream=True)"
            )

        if not hasattr(lit_api, "step") and not hasattr(lit_api, "predict"):
            raise ValueError("""Using the default step method with Continuous batching loop requires the lit_api to
have a `predict` method which accepts decoded request inputs and a list of generated_sequence.
Please implement the has_finished method in the lit_api.

    class ExampleAPI(LitAPI):
        ...
        def predict(self, inputs, generated_sequence):
            # implement predict logic
            # return list of new tokens
            ...
        """)

        if not hasattr(lit_api, "step") and not hasattr(lit_api, "has_finished"):
            raise ValueError("""Using the default step method with Continuous batching loop
requires the lit_api to have a has_finished method. Please implement the has_finished method in the lit_api.

    class ExampleAPI(LitAPI):
        ...
        def has_finished(self, uid: str, token: str, max_sequence_length: int) -> bool:
            # implement has_finished logic
            return False
        """)

    def add_request(self, uid: str, request: Any, lit_api: LitAPI, lit_spec: Optional[LitSpec]) -> None:
        """Add a new sequence to active sequences and perform any action before prediction such as filling the cache."""
        if hasattr(lit_api, "add_request"):
            lit_api.add_request(uid, request)
        decoded_request = lit_api.decode_request(request)
        self.active_sequences[uid] = {"input": decoded_request, "current_length": 0, "generated_sequence": []}

    def mark_completed(self, uid: str) -> None:
        """Mark a request as completed and remove it from the tracked state."""
        logger.debug(f"Marking sequence {uid} as completed")
        del self.active_sequences[uid]
        del self.response_queue_ids[uid]

    def has_capacity(self, lit_api: LitAPI) -> bool:
        """Check if we can add more sequences based on current batch."""
        capacity = len(self.active_sequences) < lit_api.max_batch_size
        if not capacity:
            logger.info(
                f"No capacity: {len(self.active_sequences)} active sequences, max batch size: {lit_api.max_batch_size}"
            )
        return capacity

    def step(self, prev_outputs: Optional[List[Output]], lit_api: LitAPI, lit_spec: Optional[LitSpec]) -> List[Output]:
        """Process one token generation step for all active sequences."""
        if hasattr(lit_api, "step"):
            return lit_api.step(prev_outputs)

        if not self.active_sequences:
            return []

        # Batch forward pass for all active sequences
        inputs = [seq["input"] for seq in self.active_sequences.values()]
        generated = [seq["generated_sequence"] for seq in self.active_sequences.values()]

        try:
            # Assume lit_api.predict handles batched token generation
            new_tokens: List[Any] = lit_api.predict(inputs, generated)

            responses: List[Output] = []

            # Process each sequence's new token
            for uid, token in zip(self.active_sequences.keys(), new_tokens):
                seq = self.active_sequences[uid]
                seq["generated_sequence"].append(token)
                seq["current_length"] += 1

                step_output = Output(uid, token, LitAPIStatus.OK)
                responses.append(step_output)

                # Check completion conditions
                is_finished = lit_api.has_finished(uid, token, self.max_sequence_length)

                if is_finished:
                    # Encode final response for completed sequence
                    step_output = Output(uid, "", LitAPIStatus.FINISH_STREAMING)
                    responses.append(step_output)

            return responses

        except Exception as e:
            logger.exception("Error during batch token generation")
            # On error, terminate all active sequences
            responses = [(uid, (e, LitAPIStatus.ERROR)) for uid in self.active_sequences]
            self.active_sequences.clear()
            return responses

    def prefill(
        self,
        pending_requests: List[Tuple[str, Any]],
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        request_queue: Queue,
        max_batch_size: int,
        response_queues: List[Queue],
    ) -> List[Tuple[str, Any]]:
        """Fill available capacity with pending and new requests."""
        # First process existing pending requests
        while pending_requests and self.has_capacity(lit_api):
            response_queue_id, uid, input = pending_requests.pop(0)
            self.add_request(uid, input, lit_api, lit_spec)
            self.response_queue_ids[uid] = response_queue_id

        # Then check for new requests if we still have capacity
        if self.has_capacity(lit_api):
            new_batches, timed_out_uids = self.get_batch_requests(
                lit_api, request_queue, max_batch_size, batch_timeout=0.0001
            )
            notify_timed_out_requests(response_queues, timed_out_uids)

            if new_batches:
                # Add new requests to pending_requests and try to process them
                for response_queue_id, uid, input in new_batches:
                    logger.debug(f"New request: {uid}, {input}")
                    if self.has_capacity(lit_api):
                        self.add_request(uid, input, lit_api, lit_spec)
                        self.response_queue_ids[uid] = response_queue_id
                    else:
                        pending_requests.append((response_queue_id, uid, input))

        return pending_requests

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
        """Main loop that processes batches of requests."""
        pending_requests = self.prefill(
            [],
            lit_api,
            lit_spec,
            request_queue,
            max_batch_size,
            response_queues,
        )
        try:
            prev_outputs = None
            while pending_requests or self.active_sequences:
                # Process one step for all active sequences
                responses = self.step(prev_outputs, lit_api, lit_spec)
                logger.debug(f"Responses from step(): {responses}")
                if len(responses) == 0:
                    raise HTTPException(500, "No responses from step()")
                if responses and not isinstance(responses[0], Output):
                    raise HTTPException(500, "Expected StepOutput from step()")

                prev_outputs = responses

                # Send responses for all sequences (both streaming and completed)
                for step_output in responses:
                    logger.debug(f"Processing response: {step_output}")
                    status = step_output.status
                    response_data = lit_api.encode_response(step_output.output)
                    uid = step_output.uid
                    response_queue_id = self.response_queue_ids[uid]

                    response_data = lit_api.format_encoded_response(response_data)
                    if status == LitAPIStatus.ERROR:
                        self.put_error_response(response_queues, response_queue_id, uid, response_data)
                        self.mark_completed(uid)
                    elif status == LitAPIStatus.FINISH_STREAMING:
                        self.put_response(response_queues, response_queue_id, uid, response_data, status)
                        self.mark_completed(uid)
                    else:
                        self.put_response(response_queues, response_queue_id, uid, response_data, status)

                # Fill available capacity with both pending and new requests
                pending_requests = self.prefill(
                    pending_requests,
                    lit_api,
                    lit_spec,
                    request_queue,
                    max_batch_size,
                    response_queues,
                )

        except Exception as e:
            logger.exception(f"Error in continuous batching loop: {e}")
            # Handle any errors by sending error responses for all tracked requests
            for uid, response_queue_id in self.response_queue_ids.items():
                self.put_error_response(response_queues, response_queue_id, uid, e)
            self.response_queue_ids.clear()
            self.active_sequences.clear()


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
