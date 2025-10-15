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
import os
from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.loops.base import LitLoop, _BaseLoop
from litserve.loops.simple_loops import BatchedLoop, SingleLoop
from litserve.loops.streaming_loops import BatchedStreamingLoop, StreamingLoop
from litserve.transport.base import MessageTransport
from litserve.utils import WorkerSetupStatus

logger = logging.getLogger(__name__)


def get_default_loop(stream: bool, max_batch_size: int, enable_async: bool = False) -> _BaseLoop:
    """Get the default loop based on the stream flag, batch size, and async support.

    Args:
        stream: Whether streaming is enabled
        max_batch_size: Maximum batch size
        enable_async: Whether async support is enabled (supports both coroutines and async generators)

    Returns:
        The appropriate loop implementation

    Raises:
        ValueError: If async and batching are enabled together (not supported)

    """
    if enable_async:
        if max_batch_size > 1:
            raise ValueError("Async batching is not supported. Please use enable_async=False with batching.")
        if stream:
            return StreamingLoop()  # StreamingLoop now supports async
        return SingleLoop()  # Only SingleLoop supports async currently

    if stream:
        if max_batch_size > 1:
            return BatchedStreamingLoop()
        return StreamingLoop()

    if max_batch_size > 1:
        return BatchedLoop()
    return SingleLoop()


def inference_worker(
    lit_api: LitAPI,
    device: str,
    worker_id: int,
    request_queue: Queue,
    transport: MessageTransport,
    workers_setup_status: dict[int, str],
    callback_runner: CallbackRunner,
):
    os.environ["LITSERVE_WORKER_ID"] = str(worker_id)

    lit_spec = lit_api.spec
    loop: LitLoop = lit_api.loop
    stream = lit_api.stream

    endpoint = lit_api.api_path.split("/")[-1]

    callback_runner.trigger_event(EventTypes.BEFORE_SETUP.value, lit_api=lit_api)
    try:
        lit_api.setup(device)
    except Exception:
        logger.exception(f"Error setting up worker {worker_id}.")
        workers_setup_status[f"{endpoint}_{worker_id}"] = WorkerSetupStatus.ERROR
        return
    lit_api.device = device
    callback_runner.trigger_event(EventTypes.AFTER_SETUP.value, lit_api=lit_api)

    if workers_setup_status:
        workers_setup_status[f"{endpoint}_{worker_id}"] = WorkerSetupStatus.READY

    if lit_spec:
        logging.info(f"LitServe will use {lit_spec.__class__.__name__} spec")

    if loop == "auto":
        loop = get_default_loop(stream, lit_api.max_batch_size, lit_api.enable_async)

    loop(
        lit_api,
        device,
        worker_id,
        request_queue,
        transport,
        workers_setup_status,
        callback_runner,
    )
