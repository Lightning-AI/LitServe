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
import logging
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Optional

from fastapi import HTTPException

from litserve import LitAPI
from litserve.callbacks import CallbackRunner
from litserve.loops.base import LitLoop
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.utils import LitAPIStatus, LoopResponseType

logger = logging.getLogger(__name__)


def notify_timed_out_requests(
    response_queues: list[Queue],
    timed_out_uids: list[tuple[int, str]],
):
    for response_queue_id, uid in timed_out_uids:
        logger.error(f"Request {uid} was waiting in the queue for too long and has been timed out.")
        response_queues[response_queue_id].put(
            (
                uid,
                (HTTPException(504, "Request timed out"), LitAPIStatus.ERROR, LoopResponseType.STREAMING),
            )
        )


@dataclass
class Output:
    """Outputs from a single step of the loop."""

    uid: str
    output: Any
    status: LitAPIStatus


class ContinuousBatchingLoop(LitLoop):
    def __init__(self, max_sequence_length: int = 2048, no_pending_requests: bool = False, sleep_delay: float = 0.001):
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
        self.active_sequences: dict[str, dict] = {}  # uid -> {input, current_length, generated_sequence}
        self.max_sequence_length = max_sequence_length
        self.response_queue_ids: dict[str, int] = {}  # uid -> response_queue_id
        self.no_pending_requests = no_pending_requests
        self.sleep_delay = sleep_delay

    def pre_setup(self, lit_api: LitAPI, spec: Optional[LitSpec] = None):
        """Check if the lit_api has the necessary methods and if streaming is enabled."""
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

    def add_request(
        self,
        uid: str,
        request: Any,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        transport: Optional[MessageTransport] = None,
    ) -> None:
        """Add a new sequence to active sequences and perform any action before prediction such as filling the cache."""
        lit_api.add_request(uid, request)
        self.active_sequences[uid] = {"input": request}

    def mark_completed(self, uid: str) -> None:
        """Mark a request as completed and remove it from the tracked state."""
        logger.debug(f"Marking sequence {uid} as completed")
        del self.active_sequences[uid]
        del self.response_queue_ids[uid]

    def has_capacity(self, lit_api: LitAPI) -> bool:
        """Check if we can add more sequences based on current batch."""
        return lit_api.has_capacity()

    async def prefill(
        self,
        pending_requests: list[tuple[str, Any]],
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        request_queue: Queue,
        transport: MessageTransport,
        max_batch_size: Optional[int] = None,
        batch_timeout: Optional[float] = None,
    ) -> list[tuple[str, Any]]:
        """Fill available capacity with pending and new requests."""
        # First process existing pending requests
        while pending_requests and self.has_capacity(lit_api):
            response_queue_id, uid, input = pending_requests.pop(0)
            self.add_request(uid, input, lit_api, lit_spec, transport)
            self.response_queue_ids[uid] = response_queue_id

        while True:
            if self.no_pending_requests and lit_api.has_active_requests():
                await asyncio.sleep(self.sleep_delay)
                return pending_requests

            request = await asyncio.to_thread(self.get_request, request_queue, timeout=1, block=True)
            if request is None:
                break

            response_queue_id, uid, timestamp, input = request

            logger.debug(
                f"[worker {self.worker_id}] uid:{uid}, duration:{time.monotonic() - timestamp},"
                f"pending_requests: {len(pending_requests)}"
            )

            self.put_response(
                transport=transport,
                response_queue_id=response_queue_id,
                uid=uid,
                response_data=(),
                status=LitAPIStatus.START,
                response_type=LoopResponseType.STREAMING,
            )

            if self.has_capacity(lit_api):
                logger.debug(f"New request: {uid}, {input}")
                self.response_queue_ids[uid] = response_queue_id
                self.add_request(uid, input, lit_api, lit_spec, transport)
            else:
                pending_requests.append((response_queue_id, uid, input))
                break

        return pending_requests

    async def schedule_task(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        request_queue: Queue,
        transport: MessageTransport,
    ):
        logger.info("Running prefill in background")
        try:
            pending_requests = []
            while True:
                pending_requests = await self.prefill(
                    pending_requests,
                    lit_api,
                    lit_spec,
                    request_queue,
                    transport,
                    max_batch_size=lit_api.max_batch_size,
                    batch_timeout=lit_api.batch_timeout,
                )
                await asyncio.sleep(0)
        except Exception as e:
            logger.exception("An error occurred in run_in_background: %s", e)
        finally:
            logger.info("Exiting run_in_background in continuous_batching_loop")

    async def step(
        self, prev_outputs: Optional[list[Output]], lit_api: LitAPI, lit_spec: Optional[LitSpec]
    ) -> list[Output]:
        return await asyncio.to_thread(lit_api.step, prev_outputs)

    async def run(
        self,
        lit_api: LitAPI,
        device: str,
        worker_id: int,
        request_queue: Queue,
        transport: MessageTransport,
        workers_setup_status: dict[int, str],
        callback_runner: CallbackRunner,
    ):
        """Main loop that processes batches of requests."""
        lit_spec = lit_api.spec
        try:
            prev_outputs = None
            while lit_api.has_active_requests():
                # Process one step for all active sequences
                responses = await self.step(prev_outputs, lit_api, lit_spec)
                if len(responses) == 0:
                    logger.warning("No responses from step() but has_active_requests() is true")
                    continue
                if responses and not isinstance(responses[0], Output):
                    raise HTTPException(500, "Expected StepOutput from step()")

                prev_outputs = responses
                # Send responses for all sequences (both streaming and completed)
                for step_output in responses:
                    status = step_output.status
                    response_data = lit_api.encode_response(step_output.output)
                    uid = step_output.uid
                    response_queue_id = self.response_queue_ids[uid]

                    response_data = lit_api.format_encoded_response(response_data)
                    if status == LitAPIStatus.ERROR:
                        self.put_error_response(
                            transport, response_queue_id, uid, response_data, LoopResponseType.STREAMING
                        )
                        self.mark_completed(uid)
                    elif status == LitAPIStatus.FINISH_STREAMING:
                        self.put_response(
                            transport, response_queue_id, uid, response_data, status, LoopResponseType.STREAMING
                        )
                        self.mark_completed(uid)
                    else:
                        self.put_response(
                            transport, response_queue_id, uid, response_data, status, LoopResponseType.STREAMING
                        )

        except Exception as e:
            logger.exception(f"Error in continuous batching loop: {e}")
            # Handle any errors by sending error responses for all tracked requests
            for uid, response_queue_id in self.response_queue_ids.items():
                self.put_error_response(transport, response_queue_id, uid, e, LoopResponseType.STREAMING)
            self.response_queue_ids.clear()
            self.active_sequences.clear()

    def on_schedule_task_error(self, exception: Exception):
        pass


class DefaultContinuousBatchingLoop(ContinuousBatchingLoop):
    def add_request(self, uid: str, request: Any, lit_api: LitAPI, lit_spec: Optional[LitSpec]) -> None:
        """Add a new sequence to active sequences and perform any action before prediction such as filling the cache."""
        decoded_request = lit_api.decode_request(request)
        self.active_sequences[uid] = {"input": decoded_request, "current_length": 0, "generated_sequence": []}

    async def step(
        self, prev_outputs: Optional[list[Output]], lit_api: LitAPI, lit_spec: Optional[LitSpec]
    ) -> list[Output]:
        """Process one token generation step for all active sequences."""
        if not self.active_sequences:
            return []

        # Batch forward pass for all active sequences
        inputs = [seq["input"] for seq in self.active_sequences.values()]
        generated = [seq["generated_sequence"] for seq in self.active_sequences.values()]

        try:
            # Assume lit_api.predict handles batched token generation
            new_tokens: list[Any] = lit_api.predict(inputs, generated)

            responses: list[Output] = []

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

    def has_capacity(self, lit_api: LitAPI) -> bool:
        capacity = len(self.active_sequences) < lit_api.max_batch_size
        if not capacity:
            logger.info(
                f"No capacity: {len(self.active_sequences)} active sequences, max batch size: {lit_api.max_batch_size}"
            )
        return capacity
