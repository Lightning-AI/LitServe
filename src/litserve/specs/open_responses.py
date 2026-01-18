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
from collections import deque
from typing import TYPE_CHECKING, Literal

from beartype.typing import Any, Optional
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field

from litserve.specs.base import LitSpec
from litserve.types import OpenResponseStatus, Role
from litserve.utils import LitAPIStatus, ResponseBufferItem

if TYPE_CHECKING:
    from litserve import LitAPI, LitServer

logger = logging.getLogger(__name__)


class OpenResponseRequest(BaseModel):
    model: str
    input: str | list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    stream: bool = False
    max_output_tokens: int | None = None


def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class OutputTextContent(BaseModel):
    """
    https://www.openresponses.org/reference#object-OutputTextContent-title
    """

    type: Literal["output_text"] = "output_text"
    text: str


class Message(BaseModel):
    """
    https://www.openresponses.org/reference#object-Message-title
    """

    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: generate_id("msg"))
    status: Literal["in_progress", "completed", "incomplete"]
    role: Literal["unknown", "user", "assistant", "system", "critic", "discriminator", "developer", "tool"]
    content: OutputTextContent


class OpenResponseResponse(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("openresp"))
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    completed_at: int
    status: str
    model: str
    output: list[Message] | list[dict[str, Any]]
    usage: dict[str, int] | None = None


class OpenResponsesSpec(LitSpec):
    """Spec for OpenResponses API.

    Implements the OpenResponses specification for multi-provider, interoperable LLM interfaces.
    See https://www.openresponses.org/specification for details.
    """

    def __init__(self):
        super().__init__()
        self.api_path: str = "/v1/responses"

    @property
    def stream(self):
        return True

    def pre_setup(self, lit_api: "LitAPI"):
        # register the create response endpoint
        self.add_endpoint(self.api_path, self.create_open_response, methods=["POST"])
        self.add_endpoint(self.api_path, self.options_open_responses, methods=["OPTIONS"])

        if lit_api.enable_async:
            raise ValueError("OpenResponsesSpec does not support async mode yet.")

    def setup(self, server: "LitServer"):
        super().setup(server)
        print("OpenResponses spec setup complete.")

    def populate_context(self, context, request):
        data = request.dict()
        data.pop("input")
        context.update(data)

    def decode_request(
        self, request: OpenResponseRequest, context_kwargs: Optional[dict] = None
    ) -> OpenResponseRequest:
        return request

    def encode_response(self, output_generator: Any, context_kwargs: Optional[dict] = None) -> Any:
        # distinct from LitSpec.encode_response, we handle the generator in the endpoint handlers
        return output_generator

    async def create_open_response(self, request: OpenResponseRequest):
        response_queue_id = self.response_queue_id
        logger.debug("Received open responses request %s", request)

        uid = str(uuid.uuid4())
        q = deque()
        event = asyncio.Event()

        self.response_buffer[uid] = ResponseBufferItem(response_queue=q, event=event)

        # enqueue the request
        self.request_queue.put((response_queue_id, uid, time.monotonic(), request.model_copy()))

        # wait for data streamer
        data_gen = self.data_streamer(q, event, send_status=True)

        if request.stream:
            return Response(
                status_code=400,
                content='{"error": {"message": "Streaming is not supported yet.", "type": "invalid_request_error"}}',
                media_type="application/json",
            )

        response_task = asyncio.create_task(self.non_streaming_response(request, data_gen))
        return await response_task

    async def non_streaming_response(self, request: OpenResponseRequest, data_gen: Any) -> OpenResponseResponse:
        try:
            model = request.model

            output_items = []
            text_buffer = []

            async for response, status in data_gen:
                if status == LitAPIStatus.ERROR and isinstance(response, HTTPException):
                    raise response
                if status == LitAPIStatus.ERROR:
                    logger.error("Error in streaming response: %s", response)
                    raise HTTPException(status_code=500)

                if isinstance(response, str):
                    text_buffer.append(response)

            if text_buffer:
                message = Message(
                    role=Role.ASSISTANT,
                    status=OpenResponseStatus.COMPLETED,
                    content=OutputTextContent(text="".join(text_buffer)),
                )
                output_items.append(message)

            return OpenResponseResponse(
                model=model,
                completed_at=int(time.time()),
                status=OpenResponseStatus.COMPLETED,
                output=output_items,
            )
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error("Error in non-streaming response: %s", e, exc_info=True)
            raise HTTPException(status_code=500)

    async def options_open_responses(self, request: Request):
        return Response(status_code=200)
