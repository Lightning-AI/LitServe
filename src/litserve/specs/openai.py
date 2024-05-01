import time
from typing import Literal, Optional, List, Dict, Any, Union
import uuid

from pydantic import BaseModel, Field


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


class OpenAISpec:
    def __init__(self):
        self._endpoints = [
            (chat_completion, "/v1/chat/completions", ["POST"])
        ]

        self._add_endpoint(chat_completion, "/v1/chat/completions", ["POST"])

    def _add_endpoint(path, endpoint, methods):
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints():
        return self._endpoints.copy()

    async def chat_completion(request: ChatCompletionRequest, background_tasks: BackgroundTasks) -> ChatCompletionResponse:
        # TODO here
        return ChatCompletionResponse()

    def setup(self, server):
        self._server = server
