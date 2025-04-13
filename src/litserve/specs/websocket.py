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
from typing import TYPE_CHECKING, Optional, Union

from fastapi import HTTPException, WebSocket, WebSocketDisconnect, status

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

    def decode_request(self, request: Union[str, bytes], context_kwargs: Optional[dict] = None) -> str:
        """Decode the request from the client."""
        if isinstance(request, bytes):
            return request.decode("utf-8")
        return request

    def encode_response(self, output: str, context_kwargs: Optional[dict] = None) -> bytes:
        """Encode the response to send to the client."""
        if isinstance(output, str):
            return output.encode("utf-8")
        return output

    async def ws_predict(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_json()
                response_queue_id = self.response_queue_id

                uid = uuid.uuid4()
                event = asyncio.Event()
                self._server.response_buffer[uid] = event
                # Send request to inference worker
                self._server.request_queue.put_nowait((response_queue_id, uid, time.monotonic(), payload))
                # Wait for response
                await event.wait()
                response, response_status = self._server.response_buffer.pop(uid)

                # Handle errors
                if response_status == LitAPIStatus.ERROR:
                    if isinstance(response, HTTPException):
                        await websocket.send_json({"error": response.detail, "status_code": response.status_code})
                    else:
                        await websocket.send_json({"error": "Internal server error", "status_code": 500})
                else:
                    # Send successful response back to client
                    await websocket.send_json(response)

        except WebSocketDisconnect:
            logger.debug("WebSocket cliet disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket communication: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
