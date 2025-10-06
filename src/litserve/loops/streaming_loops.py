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
from queue import Empty, Queue
from typing import Optional

from fastapi import HTTPException

from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.loops.base import _SENTINEL_VALUE, DefaultLoop, _async_inject_context, _inject_context, collate_requests
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.utils import LitAPIStatus, LoopResponseType, PickleableHTTPException

logger = logging.getLogger(__name__)


class StreamingLoop(DefaultLoop):
    def run_streaming_loop(
        self,
        lit_api: LitAPI,
        request_queue: Queue,
        transport: MessageTransport,
        callback_runner: CallbackRunner,
        lit_spec: Optional[LitSpec] = None,
    ):
        lit_spec = lit_api.spec
        while True:
            try:
                request_data = request_queue.get(timeout=1.0)
                if request_data == _SENTINEL_VALUE:
                    logger.debug("Received sentinel value, stopping loop")
                    return
                response_queue_id, uid, timestamp, x_enc = request_data
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
                    transport,
                    response_queue_id,
                    uid,
                    HTTPException(504, "Request timed out"),
                    LitAPIStatus.ERROR,
                    LoopResponseType.STREAMING,
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

                callback_runner.trigger_event(EventTypes.BEFORE_PREDICT.value, lit_api=lit_api)
                y_gen = _inject_context(
                    context,
                    lit_api.predict,
                    x,
                )
                callback_runner.trigger_event(EventTypes.AFTER_PREDICT.value, lit_api=lit_api)

                callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE.value, lit_api=lit_api)
                y_enc_gen = _inject_context(
                    context,
                    lit_api.encode_response,
                    y_gen,
                )
                for y_enc in y_enc_gen:
                    y_enc = lit_api.format_encoded_response(y_enc)
                    self.put_response(
                        transport, response_queue_id, uid, y_enc, LitAPIStatus.OK, LoopResponseType.STREAMING
                    )
                self.put_response(
                    transport, response_queue_id, uid, "", LitAPIStatus.FINISH_STREAMING, LoopResponseType.STREAMING
                )

                callback_runner.trigger_event(EventTypes.AFTER_PREDICT.value, lit_api=lit_api)
                callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE.value, lit_api=lit_api)

            except HTTPException as e:
                self.put_response(
                    transport,
                    response_queue_id,
                    uid,
                    PickleableHTTPException.from_exception(e),
                    LitAPIStatus.ERROR,
                    LoopResponseType.STREAMING,
                )
            except KeyboardInterrupt:  # pragma: no cover
                self.kill()
                return
            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                    "Please check the error trace for more details.",
                    uid,
                )
                self.put_error_response(transport, response_queue_id, uid, e, LoopResponseType.STREAMING)

    async def _process_streaming_request(
        self,
        request,
        lit_api: LitAPI,
        transport: MessageTransport,
        callback_runner: CallbackRunner,
        lit_spec: Optional[LitSpec] = None,
    ):
        lit_spec = lit_api.spec
        response_queue_id, uid, timestamp, x_enc = request
        try:
            context = {}
            if hasattr(lit_spec, "populate_context"):
                lit_spec.populate_context(context, x_enc)

            callback_runner.trigger_event(EventTypes.BEFORE_DECODE_REQUEST.value, lit_api=lit_api)
            x = await _async_inject_context(
                context,
                lit_api.decode_request,
                x_enc,
            )
            callback_runner.trigger_event(EventTypes.AFTER_DECODE_REQUEST.value, lit_api=lit_api)

            callback_runner.trigger_event(EventTypes.BEFORE_PREDICT.value, lit_api=lit_api)
            y_gen = await _async_inject_context(
                context,
                lit_api.predict,
                x,
            )
            callback_runner.trigger_event(EventTypes.AFTER_PREDICT.value, lit_api=lit_api)

            callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE.value, lit_api=lit_api)

            # When using async, predict should return an async generator
            # and encode_response should handle async generators
            # The _async_inject_context already handles async generators correctly
            enc_result = await _async_inject_context(
                context,
                lit_api.encode_response,
                y_gen,
            )

            # encode_response should also return an async generator
            async for y_enc in enc_result:
                y_enc = lit_api.format_encoded_response(y_enc)
                self.put_response(transport, response_queue_id, uid, y_enc, LitAPIStatus.OK, LoopResponseType.STREAMING)

            self.put_response(
                transport, response_queue_id, uid, "", LitAPIStatus.FINISH_STREAMING, LoopResponseType.STREAMING
            )
            callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE.value, lit_api=lit_api)

        except HTTPException as e:
            self.put_response(
                transport=transport,
                response_queue_id=response_queue_id,
                uid=uid,
                response_data=PickleableHTTPException.from_exception(e),
                status=LitAPIStatus.ERROR,
                response_type=LoopResponseType.STREAMING,
            )
        except Exception as e:
            logger.exception(
                "LitAPI ran into an error while processing the streaming request uid=%s.\n"
                "Please check the error trace for more details.",
                uid,
            )
            self.put_error_response(transport, response_queue_id, uid, e, LoopResponseType.STREAMING)

    def run_streaming_loop_async(
        self,
        lit_api: LitAPI,
        request_queue: Queue,
        transport: MessageTransport,
        callback_runner: CallbackRunner,
    ):
        if lit_api.spec:
            # wrap the default implementation of the spec in an async spec wrapper
            lit_api.spec = lit_api.spec.as_async()

        async def process_requests():
            event_loop = asyncio.get_running_loop()
            pending_tasks = set()

            while True:
                try:
                    request_data = await event_loop.run_in_executor(None, request_queue.get, 1.0)
                    if request_data == _SENTINEL_VALUE:
                        logger.debug("Received sentinel value, stopping loop")
                        return
                    response_queue_id, uid, timestamp, x_enc = request_data
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
                        transport,
                        response_queue_id,
                        uid,
                        HTTPException(504, "Request timed out"),
                        LitAPIStatus.ERROR,
                        LoopResponseType.STREAMING,
                    )
                    continue

                task = asyncio.create_task(
                    self._process_streaming_request(
                        (response_queue_id, uid, timestamp, x_enc),
                        lit_api,
                        transport,
                        callback_runner,
                    ),
                    name=f"streaming_request_{uid}",
                )
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)

        loop = asyncio.new_event_loop()

        try:
            loop.run_until_complete(process_requests())
        except KeyboardInterrupt:
            self.kill()

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
        if lit_api.enable_async:
            self.run_streaming_loop_async(lit_api, request_queue, transport, callback_runner)
        else:
            self.run_streaming_loop(lit_api, request_queue, transport, callback_runner)


class BatchedStreamingLoop(DefaultLoop):
    def run_batched_streaming_loop(
        self,
        lit_api: LitAPI,
        request_queue: Queue,
        transport: MessageTransport,
        callback_runner: CallbackRunner,
        lit_spec: Optional[LitSpec] = None,
    ):
        lit_spec = lit_api.spec
        while True:
            batches, timed_out_uids = collate_requests(
                lit_api,
                request_queue,
            )
            for response_queue_id, uid in timed_out_uids:
                logger.error(
                    f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                    "has been timed out. "
                    "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
                )
                self.put_response(
                    transport,
                    response_queue_id,
                    uid,
                    HTTPException(504, "Request timed out"),
                    LitAPIStatus.ERROR,
                    LoopResponseType.STREAMING,
                )

            if not batches:
                continue
            response_queue_ids, uids, inputs = zip(*batches)
            num_inputs = len(inputs)
            try:
                contexts = [{} for _ in range(num_inputs)]
                if hasattr(lit_spec, "populate_context"):
                    for input, context in zip(inputs, contexts):
                        lit_spec.populate_context(context, input)

                callback_runner.trigger_event(EventTypes.BEFORE_DECODE_REQUEST.value, lit_api=lit_api)
                x = [
                    _inject_context(
                        context,
                        lit_api.decode_request,
                        input,
                    )
                    for input, context in zip(inputs, contexts)
                ]
                callback_runner.trigger_event(EventTypes.AFTER_DECODE_REQUEST.value, lit_api=lit_api)

                x = lit_api.batch(x)

                callback_runner.trigger_event(EventTypes.BEFORE_PREDICT.value, lit_api=lit_api)
                y_iter = _inject_context(contexts, lit_api.predict, x)
                callback_runner.trigger_event(EventTypes.AFTER_PREDICT.value, lit_api=lit_api)

                unbatched_iter = lit_api.unbatch(y_iter)

                callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE.value, lit_api=lit_api)
                y_enc_iter = _inject_context(contexts, lit_api.encode_response, unbatched_iter)
                callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE.value, lit_api=lit_api)

                # y_enc_iter -> [[response-1, response-2], [response-1, response-2]]
                # Track which items in the batch have finished
                finished_items = set()
                
                for y_batch in y_enc_iter:
                    for idx, (response_queue_id, y_enc, uid) in enumerate(zip(response_queue_ids, y_batch, uids)):
                        # Skip items that have already finished
                        if idx in finished_items:
                            continue
                        
                        # Check if this item has finished (None indicates EOS/end of sequence)
                        if y_enc is None:
                            finished_items.add(idx)
                            # Send finish signal for this specific item
                            self.put_response(
                                transport, response_queue_id, uid, "", LitAPIStatus.FINISH_STREAMING, LoopResponseType.STREAMING
                            )
                        else:
                            y_enc = lit_api.format_encoded_response(y_enc)
                            self.put_response(
                                transport, response_queue_id, uid, y_enc, LitAPIStatus.OK, LoopResponseType.STREAMING
                            )
                
                # Send finish signal for any items that haven't finished yet
                for idx, (response_queue_id, uid) in enumerate(zip(response_queue_ids, uids)):
                    if idx not in finished_items:
                        self.put_response(
                            transport, response_queue_id, uid, "", LitAPIStatus.FINISH_STREAMING, LoopResponseType.STREAMING
                        )
            except KeyboardInterrupt:  # pragma: no cover
                self.kill()
                return

            except HTTPException as e:
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    self.put_response(
                        transport,
                        response_queue_id,
                        uid,
                        PickleableHTTPException.from_exception(e),
                        LitAPIStatus.ERROR,
                        LoopResponseType.STREAMING,
                    )

            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the streaming batched request.\n"
                    "Please check the error trace for more details."
                )
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    self.put_error_response(transport, response_queue_id, uid, e, LoopResponseType.STREAMING)

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
        self.run_batched_streaming_loop(
            lit_api,
            request_queue,
            transport,
            callback_runner,
        )
