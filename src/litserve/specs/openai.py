import time
import typing
from typing import Literal, Optional, List, Dict, Union
import uuid
from fastapi import BackgroundTasks, HTTPException
from .base import LitSpec
from pydantic import BaseModel, Field

from ..server import wait_for_queue_timeout, LitAPIStatus

if typing.TYPE_CHECKING:
    from litserve import LitServer


def shortuuid():
    return uuid.uuid4().hex[:6]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = ""
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


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


class OpenAISpec(LitSpec):
    def __init__(
        self,
    ):
        # register the endpoint
        self.add_endpoint("/v1/chat/completions", self.chat_completion, ["POST"])

    def setup(self, server: "LitServer"):
        self._server = server
        self._lit_api = server.app.lit_api

    def decode_request_hook(self, request: ChatCompletionRequest):
        return request.messages

    def encode_response_hook(self, output) -> ChatCompletionResponse:
        return output

    async def chat_completion(
        self, request: ChatCompletionRequest, background_tasks: BackgroundTasks
    ) -> ChatCompletionResponse:
        # TODO here
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming not currently supported")

        if request.stop is not None:
            raise HTTPException(status_code=400, detail="Parameter stop not currently supported")

        if request.frequency_penalty:
            raise HTTPException(status_code=400, detail="Parameter frequency_penalty not currently supported")

        if request.presence_penalty:
            raise HTTPException(status_code=400, detail="Parameter presence_penalty not currently supported")

        if request.max_tokens is not None:
            raise HTTPException(status_code=400, detail="Parameter max_tokens not currently supported")

        if request.top_p != 1.0:
            raise HTTPException(status_code=400, detail="Parameter top_p not currently supported")

        uids = [uuid.uuid4() for _ in range(request.n)]
        pipes = []
        for uid in uids:
            read, write = self._server.new_pipe()

            self._server.app.request_buffer[uid] = (request, write)
            self._server.app.request_queue.put(uid)

            # TODO: Enable background cleanup here
            # background_tasks.add_task(cleanup, self._server.app.request_buffer, uid)
            pipes.append(read)

        responses = []
        for uid, read in zip(uids, pipes):
            data = await wait_for_queue_timeout(
                self._server.data_reader(read), self._server.app.timeout, uid, self._server.app.request_buffer
            )
            responses.append(data)

        choices = []

        usage = UsageInfo()
        for i, data in enumerate(responses):
            response, status = data
            if status != LitAPIStatus.OK:
                break

            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=response["text"]),
                    finish_reason=response.get("finish_reason", "stop"),
                )
            )
            task_usage = UsageInfo.parse_obj(response["usage"]) if "usage" in response else UsageInfo()
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        model = request.model or "litserve"
        return ChatCompletionResponse(model=model, choices=choices, usage=usage)
