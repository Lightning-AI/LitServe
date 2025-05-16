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
import inspect
import json
import logging
import time
import typing
import uuid
from collections import deque
from enum import Enum
from typing import Annotated, AsyncGenerator, Dict, Iterator, List, Literal, Optional, Union

from fastapi import BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, azip

if typing.TYPE_CHECKING:
    from litserve import LitAPI, LitServer

logger = logging.getLogger(__name__)


def shortuuid():
    return uuid.uuid4().hex[:6]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

    def __add__(self, other: "UsageInfo") -> "UsageInfo":
        other.prompt_tokens += self.prompt_tokens
        other.completion_tokens += self.completion_tokens
        other.total_tokens += self.total_tokens
        return other

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)


class TextContent(BaseModel):
    type: str
    text: str


class ImageDetail(str, Enum):
    auto: str = "auto"
    low: str = "low"
    high: str = "high"


class ImageContentURL(BaseModel):
    url: str
    detail: ImageDetail = ImageDetail.auto


class ImageContent(BaseModel):
    type: str
    image_url: Union[str, ImageContentURL]


class InputAudio(BaseModel):
    data: str  # base64 encoded audio data.
    format: Literal["wav", "mp3"]


class AudioContent(BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudio


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, object]


class ToolChoice(str, Enum):
    auto: str = "auto"
    none: str = "none"
    any: str = "any"


class Tool(BaseModel):
    type: Literal["function"]
    function: Function


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    index: int = 0
    id: Optional[str] = None
    type: str = "function"
    function: FunctionCall


class ResponseFormatText(BaseModel):
    type: Literal["text"]


class ResponseFormatJSONObject(BaseModel):
    type: Literal["json_object"]


class JSONSchema(BaseModel):
    name: str
    description: Optional[str] = None
    schema_: Optional[Dict[str, object]] = Field(None, alias="schema")
    strict: Optional[bool] = False


class ResponseFormatJSONSchema(BaseModel):
    json_schema: JSONSchema
    type: Literal["json_schema"]


ResponseFormat = Annotated[
    Union[ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema], "ResponseFormat"
]


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Union[TextContent, ImageContent, AudioContent]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatMessageWithUsage(ChatMessage):
    prompt_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0


class ChoiceDelta(ChatMessage):
    content: Optional[str] = None
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = ""
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None  # Kept for backward compatibility
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = ToolChoice.auto
    response_format: Optional[ResponseFormat] = None
    metadata: Optional[Dict[str, str]] = None


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
    delta: Optional[ChoiceDelta]
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    index: int
    logprobs: Optional[dict] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[ChatCompletionStreamingChoice]
    usage: Optional[UsageInfo]


LITAPI_VALIDATION_MSG = """LitAPI.predict and LitAPI.encode_response must be a generator (use yield instead or return)
while using the OpenAISpec.

Error: {}

Please follow the below examples for guidance on how to use the spec:

If your current code looks like this:

```
import litserve as ls
from litserve.specs.openai import ChatMessage

class ExampleAPI(ls.LitAPI):
    ...
    def predict(self, x):
        return "This is a generated output"

    def encode_response(self, output: dict):
        return ChatMessage(role="assistant", content="This is a custom encoded output")
```

You should modify it to:

```
import litserve as ls
from litserve.specs.openai import ChatMessage

class ExampleAPI(ls.LitAPI):
    ...
    def predict(self, x):
        yield "This is a generated output"

    def encode_response(self, output):
        yield ChatMessage(role="assistant", content="This is a custom encoded output")
```


You can also yield responses in chunks. LitServe will handle the streaming for you:

```
class ExampleAPI(ls.LitAPI):
    ...
    def predict(self, x):
        yield from self.model(x)

    def encode_response(self, output):
        for out in output:
            yield ChatMessage(role="assistant", content=out)
```
"""

ASYNC_LITAPI_VALIDATION_MSG = """LitAPI.decode_request, LitAPI.predict, and LitAPI.encode_response must all be async
coroutines (use 'async def') while using the OpenAISpec with async enabled in LitAPI.

Additionally, LitAPI.predict and LitAPI.encode_response must be async generators (use 'yield' or 'yield from' inside
 an 'async def' function).

Error: {}

Please follow the examples below for guidance on how to use the spec in async mode:

If your current code looks like this:

```
import litserve as ls
from litserve.specs.openai import ChatMessage

class ExampleAPI(ls.LitAPI):
    ...
    def predict(self, x):
        return "This is a generated output"

    def encode_response(self, output: dict):
        return ChatMessage(role="assistant", content="This is a custom encoded output")
```
You should modify it to:

```
import litserve as ls
from litserve.specs.openai import ChatMessage

class ExampleAPI(ls.LitAPI):
    ...
    async def decode_request(self, request):
        # process request here
        return request.messages

    async def predict(self, x):
        yield "This is a generated output"

    async def encode_response(self, output):
        yield ChatMessage(role="assistant", content="This is a custom encoded output")
```
You can also yield responses in chunks. LitServe will handle the streaming for you:

```
class ExampleAPI(ls.LitAPI):
    ...
    async def decode_request(self, request):
        # process request here
        return request.messages

    async def predict(self, x):
        async for out in self.model(x):
            yield out

    async def encode_response(self, output):
        async for out in output:
            yield ChatMessage(role="assistant", content=out)
```
"""


class OpenAISpec(LitSpec):
    def __init__(
        self,
    ):
        super().__init__()
        # register the endpoint
        self.add_endpoint("/v1/chat/completions", self.chat_completion, ["POST"])
        self.add_endpoint("/v1/chat/completions", self.options_chat_completions, ["OPTIONS"])

    @property
    def stream(self):
        return True

    def pre_setup(self, lit_api: "LitAPI"):
        from litserve import LitAPI

        is_encode_response_original = lit_api.encode_response.__code__ is LitAPI.encode_response.__code__

        if lit_api.enable_async:
            if not asyncio.iscoroutinefunction(lit_api.decode_request):
                raise ValueError(ASYNC_LITAPI_VALIDATION_MSG.format("decode_request is not a coroutine"))

            if not inspect.isasyncgenfunction(lit_api.predict):
                raise ValueError(ASYNC_LITAPI_VALIDATION_MSG.format("predict is not a generator"))

            if not inspect.isasyncgenfunction(lit_api.encode_response):
                raise ValueError(ASYNC_LITAPI_VALIDATION_MSG.format("encode_response is not a generator"))

        else:
            for method in ["decode_request", "predict", "encode_response"]:
                method_obj = getattr(lit_api, method)
                if asyncio.iscoroutinefunction(method_obj) or inspect.isasyncgenfunction(method_obj):
                    raise ValueError(
                        f"'{method}' is defined as async/coroutine, but 'enable_async' is not set in LitAPI. "
                        "Please set 'enable_async=True' in your LitAPI class or "
                        "use synchronous (non-async) methods instead."
                    )

            if not inspect.isgeneratorfunction(lit_api.predict):
                raise ValueError(LITAPI_VALIDATION_MSG.format("predict is not a generator"))

            if not is_encode_response_original and not inspect.isgeneratorfunction(lit_api.encode_response):
                raise ValueError(LITAPI_VALIDATION_MSG.format("encode_response is not a generator"))

    def setup(self, server: "LitServer"):
        super().setup(server)
        print("OpenAI spec setup complete")

    def populate_context(self, context, request):
        data = request.dict()
        data.pop("messages")
        context.update(data)

    def decode_request(
        self, request: ChatCompletionRequest, context_kwargs: Optional[dict] = None
    ) -> List[Dict[str, str]]:
        # returns [{"role": "system", "content": "..."}, ...]
        return [el.model_dump(by_alias=True, exclude_none=True) for el in request.messages]

    def batch(self, inputs):
        return list(inputs)

    def unbatch(self, output):
        yield output

    def extract_usage_info(self, output: Dict) -> Dict:
        prompt_tokens: int = output.pop("prompt_tokens", 0)
        completion_tokens: int = output.pop("completion_tokens", 0)
        total_tokens: int = output.pop("total_tokens", 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def validate_chat_message(self, obj):
        return isinstance(obj, dict) and "role" in obj and "content" in obj

    def _encode_response(self, output: Union[Dict[str, str], List[Dict[str, str]]]) -> Dict:
        logger.debug(output)
        if output is None:
            message = {"role": "assistant", "content": None}
        elif isinstance(output, str):
            message = {"role": "assistant", "content": output}
        elif self.validate_chat_message(output):
            message = output
        elif isinstance(output, dict) and "content" in output:
            message = output.copy()
            message.update(role="assistant")
        elif isinstance(output, list) and output and self.validate_chat_message(output[-1]):
            message = output[-1]
        else:
            error = (
                "Malformed output from LitAPI.predict: expected "
                f"string or {{'role': '...', 'content': '...'}}, got '{output}'."
            )
            logger.error(error)
            raise HTTPException(500, error)
        usage_info = self.extract_usage_info(message)
        return {**message, **usage_info}

    def encode_response(
        self, output_generator: Union[Dict[str, str], List[Dict[str, str]]], context_kwargs: Optional[dict] = None
    ) -> Iterator[Union[ChatMessage, ChatMessageWithUsage]]:
        for output in output_generator:
            logger.debug(output)
            yield self._encode_response(output)

    async def get_from_queues(self, uids) -> List[AsyncGenerator]:
        choice_pipes = []
        for uid, q, event in zip(uids, self.queues, self.events):
            data = self._server.data_streamer(q, event, send_status=True)
            choice_pipes.append(data)
        return choice_pipes

    async def options_chat_completions(self, request: Request):
        return Response(status_code=200)

    async def chat_completion(self, request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        response_queue_id = self.response_queue_id
        logger.debug("Received chat completion request %s", request)
        uids = [uuid.uuid4() for _ in range(request.n)]
        self.queues = []
        self.events = []
        for uid in uids:
            request_el = request.model_copy()
            request_el.n = 1
            q = deque()
            event = asyncio.Event()
            self._server.response_buffer[uid] = (q, event)
            self._server.request_queue.put((response_queue_id, uid, time.monotonic(), request_el))
            self.queues.append(q)
            self.events.append(event)

        responses = await self.get_from_queues(uids)

        if request.stream:
            return StreamingResponse(
                self.streaming_completion(request, responses),
                media_type="text/event-stream",
                background=background_tasks,
            )

        response_task = asyncio.create_task(self.non_streaming_completion(request, responses))
        return await response_task

    async def streaming_completion(self, request: ChatCompletionRequest, pipe_responses: List):
        model = request.model
        usage_info = None
        async for streaming_response in azip(*pipe_responses):
            choices = []
            usage_infos = []
            # iterate over n choices
            for i, (response, status) in enumerate(streaming_response):
                if status == LitAPIStatus.ERROR and isinstance(response, HTTPException):
                    raise response
                elif status == LitAPIStatus.ERROR:
                    logger.error("Error in streaming response: %s", response)
                    raise HTTPException(status_code=500)
                encoded_response = json.loads(response)
                logger.debug(encoded_response)
                chat_msg = ChoiceDelta(**encoded_response)
                usage_infos.append(UsageInfo(**encoded_response))
                choice = ChatCompletionStreamingChoice(
                    index=i, delta=chat_msg, system_fingerprint="", finish_reason=None
                )

                choices.append(choice)

            # Only use the last item from encode_response
            usage_info = sum(usage_infos)
            chunk = ChatCompletionChunk(model=model, choices=choices, usage=None)
            logger.debug(chunk)
            yield f"data: {chunk.model_dump_json(by_alias=True)}\n\n"

        choices = [
            ChatCompletionStreamingChoice(
                index=i,
                delta=ChoiceDelta(),
                finish_reason="stop",
            )
            for i in range(request.n)
        ]
        last_chunk = ChatCompletionChunk(
            model=model,
            choices=choices,
            usage=usage_info,
        )
        yield f"data: {last_chunk.model_dump_json(by_alias=True)}\n\n"
        yield "data: [DONE]\n\n"

    async def non_streaming_completion(self, request: ChatCompletionRequest, generator_list: List[AsyncGenerator]):
        model = request.model
        usage_infos = []
        choices = []
        # iterate over n choices
        for i, streaming_response in enumerate(generator_list):
            msgs = []
            tool_calls = None
            usage = None
            async for response, status in streaming_response:
                if status == LitAPIStatus.ERROR and isinstance(response, HTTPException):
                    raise response
                if status == LitAPIStatus.ERROR:
                    logger.error("Error in OpenAI non-streaming response: %s", response)
                    raise HTTPException(status_code=500)

                # data from LitAPI.encode_response
                encoded_response = json.loads(response)
                logger.debug(encoded_response)
                chat_msg = ChatMessage(**encoded_response)
                usage = UsageInfo(**encoded_response)
                usage_infos.append(usage)  # Aggregate usage info across all choices
                msgs.append(chat_msg.content)
                if chat_msg.tool_calls:
                    tool_calls = chat_msg.tool_calls

            content = "".join(msg for msg in msgs if msg is not None)
            msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
            choice = ChatCompletionResponseChoice(index=i, message=msg, finish_reason="stop")
            choices.append(choice)

        return ChatCompletionResponse(model=model, choices=choices, usage=sum(usage_infos))
