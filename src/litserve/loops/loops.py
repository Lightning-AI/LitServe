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
import logging
from queue import Queue
from typing import Dict, Optional, Union

from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.loops.base import _BaseLoop
from litserve.loops.simple_loops import BatchedLoop, SingleLoop
from litserve.loops.streaming_loops import BatchedStreamingLoop, StreamingLoop
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.utils import WorkerSetupStatus

logger = logging.getLogger(__name__)


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
    transport: MessageTransport,
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
        transport,
        max_batch_size,
        batch_timeout,
        stream,
        workers_setup_status,
        callback_runner,
    )
