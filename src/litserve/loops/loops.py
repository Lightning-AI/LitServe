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
import inspect
import logging
from queue import Queue
from typing import Dict, List, Optional, Union

from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.loops.base import (
    LitLoop,
    _BaseLoop,
    run_batched_loop,
    run_batched_streaming_loop,
    run_single_loop,
    run_streaming_loop,
)
from litserve.specs.base import LitSpec
from litserve.utils import WorkerSetupStatus

logger = logging.getLogger(__name__)


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
