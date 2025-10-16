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
import os
import pickle
import signal
import sys
import time
from abc import ABC
from queue import Empty, Queue
from typing import Any, Optional, Union

from starlette.formparsers import MultiPartParser

from litserve import LitAPI
from litserve.callbacks import CallbackRunner
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.utils import LitAPIStatus, LoopResponseType

logger = logging.getLogger(__name__)
# FastAPI writes form files to disk over 1MB by default, which prevents serialization by multiprocessing
MultiPartParser.max_file_size = sys.maxsize
# renamed in PR: https://github.com/encode/starlette/pull/2780
MultiPartParser.spool_max_size = sys.maxsize

_DEFAULT_STOP_LOOP_MESSAGE = "Received sentinel value, stopping loop"
_SENTINEL_VALUE = (None, None, None, None)


def _inject_context(context: Union[list[dict], dict], func, *args, **kwargs):
    sig = inspect.signature(func)
    if "context" in sig.parameters:
        return func(*args, **kwargs, context=context)
    return func(*args, **kwargs)


async def _sync_fn_to_async_fn(func, *args, **kwargs):
    if inspect.isgeneratorfunction(func):

        async def async_fn(*args, **kwargs):
            for item in func(*args, **kwargs):
                yield item
            return

        return async_fn(*args, **kwargs)

    return await asyncio.to_thread(func, *args, **kwargs)


async def _handle_async_function(func, *args, **kwargs):
    # Call the function based on its type
    if inspect.isasyncgenfunction(func):
        # Async generator - return directly (don't await)
        return func(*args, **kwargs)
    if asyncio.iscoroutinefunction(func):
        # Async function - await the result
        return await func(*args, **kwargs)

    # Sync function - convert to async function, then await if result is awaitable
    result = await _sync_fn_to_async_fn(func, *args, **kwargs)

    # Check if the result is awaitable (coroutine)
    if asyncio.iscoroutine(result):
        return await result

    return result


async def _async_inject_context(context: Union[list[dict], dict], func, *args, **kwargs):
    sig = inspect.signature(func)

    # Determine if we need to inject context
    if "context" in sig.parameters:
        kwargs["context"] = context

    return await _handle_async_function(func, *args, **kwargs)


class _StopLoopError(Exception):
    def __init__(self, message: str = _DEFAULT_STOP_LOOP_MESSAGE):
        self.message = message
        super().__init__(self.message)


def collate_requests(
    loop: "LitLoop",
    lit_api: LitAPI,
    request_queue: Queue,
    transport: MessageTransport,
) -> tuple[list, list]:
    payloads = []
    timed_out_uids = []
    entered_at = time.monotonic()
    end_time = entered_at + lit_api.batch_timeout
    apply_timeout = lit_api.request_timeout not in (-1, False)

    if lit_api.batch_timeout == 0:
        while len(payloads) < lit_api.max_batch_size:
            try:
                request_data = request_queue.get_nowait()
                if request_data == _SENTINEL_VALUE:
                    raise _StopLoopError()

                response_queue_id, uid, timestamp, x_enc = request_data

                loop.put_response(
                    transport=transport,
                    response_queue_id=response_queue_id,
                    uid=uid,
                    response_data=(),
                    status=LitAPIStatus.START,
                    response_type=LoopResponseType.STREAMING if lit_api.stream else LoopResponseType.REGULAR,
                )

                if apply_timeout and time.monotonic() - timestamp > lit_api.request_timeout:
                    timed_out_uids.append((response_queue_id, uid))
                else:
                    payloads.append((response_queue_id, uid, x_enc))
            except Empty:
                break
        return payloads, timed_out_uids

    while time.monotonic() < end_time and len(payloads) < lit_api.max_batch_size:
        remaining_time = end_time - time.monotonic()
        if remaining_time <= 0:
            break

        try:
            request_data = request_queue.get(timeout=min(remaining_time, 0.001))
            if request_data == _SENTINEL_VALUE:
                raise _StopLoopError()

            response_queue_id, uid, timestamp, x_enc = request_data

            loop.put_response(
                transport=transport,
                response_queue_id=response_queue_id,
                uid=uid,
                response_data=(),
                status=LitAPIStatus.START,
                response_type=LoopResponseType.STREAMING if lit_api.stream else LoopResponseType.REGULAR,
            )

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
            response_queues: list[Queue],
            stream: bool,
            workers_setup_status: dict[int, str],
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
            response_queues[response_queue_id].put((uid, (response_enc, LitAPIStatus.OK, LoopResponseType.REGULAR)))
    ```

    """

    def pre_setup(self, lit_api: LitAPI, spec: Optional[LitSpec] = None):
        pass

    async def schedule_task(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        request_queue: Queue,
        transport: MessageTransport,
    ):
        pass

    def __call__(
        self,
        lit_api: LitAPI,
        device: str,
        worker_id: int,
        request_queue: Queue,
        transport: MessageTransport,
        workers_setup_status: dict[int, str],
        callback_runner: CallbackRunner,
    ):
        lit_spec = lit_api.spec
        if asyncio.iscoroutinefunction(self.run):
            event_loop = asyncio.new_event_loop()

            async def _wrapper():
                logger.info("Running LitLoop in a asyncio event loop")
                future = self.schedule_task(lit_api, lit_spec, request_queue, transport)
                schedule_task = event_loop.create_task(future)
                while True:
                    try:
                        await self.run(
                            lit_api,
                            device,
                            worker_id,
                            request_queue,
                            transport,
                            workers_setup_status,
                            callback_runner,
                        )
                        await asyncio.sleep(0)
                    except Exception as e:
                        logger.exception("An error occurred in the loop: %s", e)

                    if not lit_api.has_active_requests() and schedule_task.done():
                        self.on_schedule_task_done(schedule_task)

            event_loop.run_until_complete(_wrapper())
        else:
            while True:
                self.run(
                    lit_api,
                    device,
                    worker_id,
                    request_queue,
                    transport,
                    workers_setup_status,
                    callback_runner,
                )

    def run(
        self,
        lit_api: LitAPI,
        device: str,
        worker_id: int,
        request_queue: Queue,
        transport: MessageTransport,
        workers_setup_status: dict[int, str],
        callback_runner: CallbackRunner,
    ):
        raise NotImplementedError

    def on_schedule_task_done(self, schedule_task: asyncio.Task) -> None:
        pass


class LitLoop(_BaseLoop):
    def __init__(self):
        self._context = {}
        self._server_pid = os.getpid()
        self._worker_id = None
        self._restart_workers = False

    def kill(self):
        try:
            logger.debug(f"Stop Server Requested - Kill parent pid [{self._server_pid}] from [{os.getpid()}]")
            if sys.platform == "win32":
                os.kill(self._server_pid, signal.SIGTERM)
        except PermissionError:
            # Access Denied because pid already killed...
            return

    def get_batch_requests(
        self,
        lit_api: LitAPI,
        request_queue: Queue,
        transport: MessageTransport,
    ) -> tuple[list, list]:
        batches, timed_out_uids = collate_requests(
            loop=self,
            lit_api=lit_api,
            request_queue=request_queue,
            transport=transport,
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

    @property
    def worker_id(self) -> Optional[int]:
        if self._worker_id is None:
            worker_id = os.environ.get("LITSERVE_WORKER_ID", None)
            self._worker_id = int(worker_id) if worker_id is not None else worker_id
        return self._worker_id

    def put_response(
        self,
        transport: MessageTransport,
        response_queue_id: int,
        uid: str,
        response_data: Any,
        status: LitAPIStatus,
        response_type: LoopResponseType,
    ) -> None:
        # Skip sending the start status if we dont plan to restart the workers
        if status == LitAPIStatus.START and not self._restart_workers:
            return

        transport.send((uid, (response_data, status, response_type, self.worker_id)), consumer_id=response_queue_id)

    def put_error_response(
        self,
        transport: MessageTransport,
        response_queue_id: int,
        uid: str,
        error: Exception,
        response_type: LoopResponseType = LoopResponseType.REGULAR,
    ) -> None:
        error = pickle.dumps(error)
        self.put_response(transport, response_queue_id, uid, error, LitAPIStatus.ERROR, response_type)


class DefaultLoop(LitLoop):
    def pre_setup(self, lit_api: LitAPI, spec: Optional[LitSpec] = None):
        # we will sanitize regularly if no spec
        # in case, we have spec then:
        # case 1: spec implements a streaming API
        # Case 2: spec implements a non-streaming API
        if lit_api.spec:
            # TODO: Implement sanitization
            return

        original = lit_api.unbatch.__code__ is LitAPI.unbatch.__code__
        if not lit_api.stream and any(
            [
                inspect.isgeneratorfunction(lit_api.predict) or inspect.isasyncgenfunction(lit_api.predict),
                inspect.isgeneratorfunction(lit_api.encode_response)
                or inspect.isasyncgenfunction(lit_api.encode_response),
            ]
        ):
            raise ValueError(
                """When `stream=False`, `lit_api.predict`, `lit_api.encode_response` must not be
                generator or async generator functions.

                Correct usage:

                    def predict(self, inputs):
                        ...
                        return {"output": output}

                    # Or async version if using LitAPI(..., enable_async=True)
                    async def predict(self, inputs):
                        ...
                        return {"output": output}

                Incorrect usage:

                    def predict(self, inputs):
                        ...
                        for i in range(max_token_length):
                            yield prediction

                    # Or async version if using LitAPI(..., enable_async=True)
                    async def predict(self, inputs):
                        ...
                        for i in range(max_token_length):
                            yield prediction
                """
            )
        if (
            lit_api.stream
            and lit_api.max_batch_size > 1
            and not all(
                [
                    inspect.isgeneratorfunction(lit_api.predict) or inspect.isasyncgenfunction(lit_api.predict),
                    inspect.isgeneratorfunction(lit_api.encode_response)
                    or inspect.isasyncgenfunction(lit_api.encode_response),
                    (
                        original
                        or inspect.isgeneratorfunction(lit_api.unbatch)
                        or inspect.isasyncgenfunction(lit_api.unbatch)
                    ),
                ]
            )
        ):
            raise ValueError(
                """When `stream=True` with max_batch_size > 1, `lit_api.predict`, `lit_api.encode_response` and
                `lit_api.unbatch` must generate values using `yield` (can be regular or async generators).

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

                # Or using async generators if using LitAPI(..., enable_async=True):
                async def predict(self, inputs):
                    ...
                    for i in range(max_token_length):
                        await asyncio.sleep(0.01)  # Some async work
                        yield prediction
             """
            )

        if lit_api.stream and not all(
            [
                inspect.isgeneratorfunction(lit_api.predict) or inspect.isasyncgenfunction(lit_api.predict),
                inspect.isgeneratorfunction(lit_api.encode_response)
                or inspect.isasyncgenfunction(lit_api.encode_response),
            ]
        ):
            raise ValueError(
                """When `stream=True` both `lit_api.predict` and
             `lit_api.encode_response` must generate values using `yield` (can be regular or async generators).

             Example:

                def predict(self, inputs):
                    ...
                    for i in range(max_token_length):
                        yield prediction

                def encode_response(self, outputs):
                    for output in outputs:
                        encoded_output = ...
                        yield encoded_output

                # Or using async generators:
                async def predict(self, inputs):
                    ...
                    for i in range(max_token_length):
                        await asyncio.sleep(0.01)  # Some async work
                        yield prediction
             """
            )
