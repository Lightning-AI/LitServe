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
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect, status

from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus

if TYPE_CHECKING:
    from litserve import LitServer

logger = logging.getLogger(__name__)


class WebSocketSpec(LitSpec):
    def __init__(self, api_path: str = "/predict"):
        super().__init__()

        # register the websocket endpoint
        self.add_ws_endpoint(api_path, self.ws_predict)

    def setup(self, server: "LitServer"):
        super().setup(server)

        print("WebSocket Spec is ready.")

    def decode_request(self, request: Dict, context_kwargs: Optional[dict] = None) -> Any:
        return request

    def encode_response(self, output: Any, context_kwargs: Optional[dict] = None) -> Dict[str, Any]:
        return output

    async def ws_predict(self, websocket: WebSocket):
        # TODO: Determine if a dedicated connection manager is needed to effectively maintain active connections
        await websocket.accept()
        response_queue_id = self.response_queue_id
        logger.debug("Received WebSocket connection: %s", websocket.client)
        try:
            while True:
                # TODO: Discuss support for additional payload formats beyond JSON.
                payload = await websocket.receive_json()

                uid = uuid.uuid4()
                event = asyncio.Event()
                self._server.response_buffer[uid] = event
                # Send request to inference worker
                self._server.request_queue.put_nowait((response_queue_id, uid, time.monotonic(), payload))
                # Wait for the response
                await event.wait()
                response, response_status = self._server.response_buffer.pop(uid)

                # Handle errors
                if response_status == LitAPIStatus.ERROR:
                    logger.error("Error in WebSocket communication: %s", response)
                    raise Exception("Error in WebSocket communication")

                logger.debug(response)

                if not isinstance(response, dict):
                    raise ValueError(
                        f"Expected response to be a dictionary, but got type {type(response)}.",
                        "The response should be a dictionary to ensure proper compatibility with the WebSocketSpec.",
                        "Please ensure that your response is a dictionary.",
                    )

                # Send successful response back to client
                await websocket.send_json(response)

        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected")
        except Exception as e:
            logger.exception("Error in WebSocket communication", exc_info=e)
            # TODO: Catch unsupported payload formats and send error message
            await websocket.send_json({"error": "Internal server error", "details": str(e)})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        finally:
            await websocket.close()
            logger.debug("WebSocket connection closed")
