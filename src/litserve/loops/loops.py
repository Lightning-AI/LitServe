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
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from litserve import LitAPI
from litserve.callbacks import CallbackRunner
from litserve.loops.base import LitLoop
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus

logger = logging.getLogger(__name__)


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
    def __init__(self, prefill_after_n_steps: int = 10, max_sequence_length: int = 2048):
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
        self.prefill_after_n_steps = prefill_after_n_steps
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
        batch_timeout: float,
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
            new_batches = []
            while True:
                batch = self.get_request(request_queue, timeout=0, block=False)
                if batch:
                    response_queue_id, uid, _, x_enc = batch
                    new_batches.append((response_queue_id, uid, x_enc))
                else:
                    break

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
            batch_timeout,
            response_queues,
        )
        try:
            prev_outputs = None
            n_steps = 0
            while pending_requests or self.active_sequences:
                # Process one step for all active sequences
                responses = self.step(prev_outputs, lit_api, lit_spec)
                logger.debug(f"Responses from step(): {responses}")
                if len(responses) == 0:
                    raise HTTPException(500, "No responses from step()")
                if responses and not isinstance(responses[0], Output):
                    raise HTTPException(500, "Expected StepOutput from step()")

                prev_outputs = responses
                n_steps += 1
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
                if n_steps > self.prefill_after_n_steps:
                    pending_requests = self.prefill(
                        pending_requests,
                        lit_api,
                        lit_spec,
                        request_queue,
                        max_batch_size,
                        batch_timeout,
                        response_queues,
                    )
                    n_steps = 0

        except Exception as e:
            logger.exception(f"Error in continuous batching loop: {e}")
            # Handle any errors by sending error responses for all tracked requests
            for uid, response_queue_id in self.response_queue_ids.items():
                self.put_error_response(response_queues, response_queue_id, uid, e)
            self.response_queue_ids.clear()
            self.active_sequences.clear()
