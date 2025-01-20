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

from starlette.formparsers import MultiPartParser

from litserve import LitAPI
from litserve.callbacks import CallbackRunner
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.utils import LitAPIStatus

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

    async def schedule_task(
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
        transport: MessageTransport,
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        if asyncio.iscoroutinefunction(self.run):
            event_loop = asyncio.new_event_loop()

            async def _wrapper():
                logger.info("Running LitLoop in a asyncio event loop")
                future = self.schedule_task(lit_api, lit_spec, request_queue, max_batch_size, batch_timeout, transport)
                _ = event_loop.create_task(future)
                while True:
                    try:
                        await self.run(
                            lit_api,
                            lit_spec,
                            device,
                            worker_id,
                            request_queue,
                            transport,
                            max_batch_size,
                            batch_timeout,
                            stream,
                            workers_setup_status,
                            callback_runner,
                        )
                        await asyncio.sleep(0)
                    except Exception as e:
                        logger.exception("An error occurred in the loop: %s", e)

            event_loop.run_until_complete(_wrapper())
        else:
            while True:
                self.run(
                    lit_api,
                    lit_spec,
                    device,
                    worker_id,
                    request_queue,
                    transport,
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
        transport: MessageTransport,
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
        self, transport: MessageTransport, response_queue_id: int, uid: str, response_data: Any, status: LitAPIStatus
    ) -> None:
        transport.send((uid, (response_data, status)), consumer_id=response_queue_id)

    def put_error_response(
        self, transport: MessageTransport, response_queue_id: int, uid: str, error: Exception
    ) -> None:
        self.put_response(transport, response_queue_id, uid, error, LitAPIStatus.ERROR)


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
        if not lit_api.stream and any([
            inspect.isgeneratorfunction(lit_api.predict),
            inspect.isgeneratorfunction(lit_api.encode_response),
        ]):
            raise ValueError(
                """When `stream=False`, `lit_api.predict`, `lit_api.encode_response` must not be
                generator functions.

                Correct usage:

                    def predict(self, inputs):
                        ...
                        return {"output": output}

                Incorrect usage:

                    def predict(self, inputs):
                        ...
                        for i in range(max_token_length):
                            yield prediction
                """
            )
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
