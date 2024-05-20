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
import time
from typing import Literal, Optional, List, Dict, Union, Generator
import uuid
from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import logging
import sys
import asyncio

from ..utils import wait_for_queue_timeout, LitAPIStatus, load_and_raise
from .base import LitSpec

logger = logging.getLogger(__name__)

PREDICT_BATCH_RESPONSE_TYPE = List[Dict[str, str]]
PREDICT_WO_BACTH_RESPONSE_TYPE = Dict[str, str]
PREDICT_RESPONSE_TYPE = Generator[Union[PREDICT_BATCH_RESPONSE_TYPE, PREDICT_WO_BACTH_RESPONSE_TYPE], None, None]


def shortuuid():
    return uuid.uuid4().hex[:6]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = ""
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class Delta(BaseModel):
    role: str
    content: str


class StreamingChoice(BaseModel):
    index: int
    delta: Delta
    logprobs: Optional[dict]
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[StreamingChoice]


class OpenAISpec(LitSpec):
    def __init__(
        self,
    ):
        super().__init__()
        # register the endpoint
        self.add_endpoint("/v1/chat/completions", self.chat_completion, ["POST"])

    def decode_request(self, request: ChatCompletionRequest) -> List[Dict[str, str]]:
        # returns [{"role": "system", "content": "..."}, ...]
        return [el.dict() for el in request.messages]

    def batch(self, inputs):
        return list(inputs)

    def unbatch(self, output_generator):
        yield from output_generator

    async def get_from_pipe(self, uids, pipes) -> List[str]:
        responses = []
        for uid, read in zip(uids, pipes):
            if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
                data = await wait_for_queue_timeout(
                    asyncio.to_thread(self._server.get_from_pipe, read),
                    self._server.timeout,
                    uid,
                    self._server.request_buffer,
                )
            else:
                data = await wait_for_queue_timeout(
                    self._server.data_reader(read), self._server.timeout, uid, self._server.request_buffer
                )
            responses.append(data)
        return responses

    async def chat_completion(
        self, request: ChatCompletionRequest, background_tasks: BackgroundTasks
    ) -> ChatCompletionResponse:
        logger.debug("Received chat completion request %s", request)
        if request.stream:
            raise HTTPException(400, "Stream is not supported")

        uids = [uuid.uuid4() for _ in range(request.n)]
        pipes = []
        for uid in uids:
            read, write = self._server.new_pipe()

            request_el = request.copy()
            request_el.n = 1
            self._server.request_buffer[uid] = (request_el, write)
            self._server.request_queue.put(uid)

            background_tasks.add_task(self._server.cleanup_request, self._server.request_buffer, uid)
            pipes.append(read)

        responses = await self.get_from_pipe(uids, pipes)

        choices = []

        usage = UsageInfo()
        for i, data in enumerate(responses):
            response, status = data
            logger.debug("Received chat completion response %s with status %s", data, status)
            if status == LitAPIStatus.ERROR:
                load_and_raise(response)

            if status != LitAPIStatus.OK:
                break

            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=response),
                    finish_reason="stop",
                )
            )
            task_usage = UsageInfo.parse_obj(UsageInfo())
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        model = request.model or "litserve"
        return ChatCompletionResponse(model=model, choices=choices, usage=usage)

    def encode_response(self, output_generator: PREDICT_RESPONSE_TYPE) -> Generator[ChatCompletionResponse, None, None]:
        yield from output_generator
