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
import json
import time
from typing import Literal, Optional, List, Dict, Union, AsyncGenerator
import uuid
from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import logging
import sys
from fastapi.responses import StreamingResponse

from .base import LitSpec

logger = logging.getLogger(__name__)


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


class ChatCompletionStreamingChoice(BaseModel):
    index: int
    delta: ChatMessage
    logprobs: Optional[dict] = None
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str
    choices: List[ChatCompletionStreamingChoice]
    usage: Optional[UsageInfo]


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

    def unbatch(self, output):
        yield output

    def validate_chat_message(self, obj):
        return isinstance(obj, dict) and "role" in obj and "content" in obj

    def _encode_response(self, output: Union[Dict[str, str], List[Dict[str, str]]]) -> ChatCompletionStreamingChoice:
        logger.debug(output)
        if isinstance(output, str):
            message = {"role": "assistant", "content": output}
        elif isinstance(output, dict) and "content" in output:
            message = output.copy()
            message.update(role="assistant")
        elif self.validate_chat_message(output):
            message = output
        elif isinstance(output, list) and output and self.validate_chat_message(output[-1]):
            message = output[-1]
        else:
            error = (
                "Malformed output from LitAPI.predict: expected"
                f"string or {{'role': '...', 'content': '...'}}, got '{output}'."
            )
            logger.exception(error)
            raise HTTPException(500, error)

        return ChatCompletionStreamingChoice(
            index=0,
            delta=ChatMessage(**message),
            finish_reason="stop",
        )

    def encode_response(
        self, output_generator: Union[Dict[str, str], List[Dict[str, str]]]
    ) -> ChatCompletionStreamingChoice:
        for output in output_generator:
            logger.info(output)
            yield self._encode_response(output)

    async def get_from_pipes(self, uids, pipes) -> List[AsyncGenerator]:
        choice_pipes = []
        for uid, (read, write) in zip(uids, pipes):
            if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
                data = self._server.win_data_streamer(read, write)
            else:
                data = self._server.data_streamer(read, write)

            choice_pipes.append(data)
        return choice_pipes

    async def chat_completion(self, request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        logger.debug("Received chat completion request %s", request)

        uids = [uuid.uuid4() for _ in range(request.n)]
        pipes = []
        for uid in uids:
            read, write = self._server.new_pipe()
            request_el = request.model_copy()
            request_el.n = 1
            self._server.request_buffer[uid] = (request_el, write)
            self._server.request_queue.put(uid)
            background_tasks.add_task(self._server.cleanup_request, self._server.request_buffer, uid)
            pipes.append((read, write))

        responses = await self.get_from_pipes(uids, pipes)
        for read, write in pipes:
            self._server.dispose_pipe(read, write)

        if request.stream:
            return StreamingResponse(self.streaming_completion(request, responses))

        return await self.non_streaming_completion(request, responses)

    async def streaming_completion(self, request: ChatCompletionRequest, pipe_responses: List):
        model = request.model
        usage = None
        for i, streaming_response in enumerate(pipe_responses):
            choices = []
            async for choice in streaming_response:
                choice = json.loads(choice)
                logger.debug(choice)
                choice = ChatCompletionStreamingChoice(**choice)
                choice.index = i
                choices.append(choice)

            yield ChatCompletionChunk(model=model, choices=choices, usage=usage, system_fingerprint="").json()

    async def non_streaming_completion(self, request: ChatCompletionRequest, pipe_responses: List):
        model = request.model
        usage = UsageInfo()
        choices = []
        for i, streaming_response in enumerate(pipe_responses):
            msgs = ""
            async for choice in streaming_response:
                choice = json.loads(choice)
                choice = ChatCompletionStreamingChoice(**choice)
                logger.debug(choice)
                # Is " " correct choice to concat with?
                if msgs:
                    msgs += " " + choice.delta.content
                else:
                    msgs = choice.delta.content

            msg = {"role": "assistant", "content": msgs}
            choice = ChatCompletionResponseChoice(index=i, message=msg, finish_reason="stop")
            choices.append(choice)

        return ChatCompletionResponse(model=model, choices=choices, usage=usage)
