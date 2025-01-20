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
import time
from queue import Empty, Queue
from typing import Dict, Optional

from fastapi import HTTPException

from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.loops.base import DefaultLoop, _inject_context, collate_requests
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.utils import LitAPIStatus, PickleableHTTPException

logger = logging.getLogger(__name__)


class StreamingLoop(DefaultLoop):
    def run_streaming_loop(
        self,
        lit_api: LitAPI,
        lit_spec: LitSpec,
        request_queue: Queue,
        transport: MessageTransport,
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
                self.put_response(
                    transport, response_queue_id, uid, HTTPException(504, "Request timed out"), LitAPIStatus.ERROR
                )
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

                callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE, lit_api=lit_api)
                y_enc_gen = _inject_context(
                    context,
                    lit_api.encode_response,
                    y_gen,
                )
                for y_enc in y_enc_gen:
                    y_enc = lit_api.format_encoded_response(y_enc)
                    self.put_response(transport, response_queue_id, uid, y_enc, LitAPIStatus.OK)
                self.put_response(transport, response_queue_id, uid, "", LitAPIStatus.FINISH_STREAMING)

                callback_runner.trigger_event(EventTypes.AFTER_PREDICT, lit_api=lit_api)
                callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE, lit_api=lit_api)

            except HTTPException as e:
                self.put_response(
                    transport,
                    response_queue_id,
                    uid,
                    PickleableHTTPException.from_exception(e),
                    LitAPIStatus.ERROR,
                )
            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                    "Please check the error trace for more details.",
                    uid,
                )
                self.put_response(transport, response_queue_id, uid, e, LitAPIStatus.ERROR)

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
        self.run_streaming_loop(lit_api, lit_spec, request_queue, transport, callback_runner)


class BatchedStreamingLoop(DefaultLoop):
    def run_batched_streaming_loop(
        self,
        lit_api: LitAPI,
        lit_spec: LitSpec,
        request_queue: Queue,
        transport: MessageTransport,
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
                self.put_response(
                    transport, response_queue_id, uid, HTTPException(504, "Request timed out"), LitAPIStatus.ERROR
                )

            if not batches:
                continue
            response_queue_ids, uids, inputs = zip(*batches)
            num_inputs = len(inputs)
            try:
                contexts = [{}] * num_inputs
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
                        self.put_response(transport, response_queue_id, uid, y_enc, LitAPIStatus.OK)

                for response_queue_id, uid in zip(response_queue_ids, uids):
                    self.put_response(transport, response_queue_id, uid, "", LitAPIStatus.FINISH_STREAMING)

            except HTTPException as e:
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    self.put_response(
                        transport,
                        response_queue_id,
                        uid,
                        PickleableHTTPException.from_exception(e),
                        LitAPIStatus.ERROR,
                    )

            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the streaming batched request.\n"
                    "Please check the error trace for more details."
                )
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    self.put_response(transport, response_queue_id, uid, e, LitAPIStatus.ERROR)

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
        self.run_batched_streaming_loop(
            lit_api,
            lit_spec,
            request_queue,
            transport,
            max_batch_size,
            batch_timeout,
            callback_runner,
        )
