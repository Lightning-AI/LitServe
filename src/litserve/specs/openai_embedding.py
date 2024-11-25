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
import logging
import time
import uuid
from typing import List, Literal, Optional, Union

from fastapi import Request, Response, status
from pydantic import BaseModel

from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus

logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    dimensions: Optional[int] = None
    encoding_format: Literal["float"] = "float"


class Embedding(BaseModel):
    index: int
    embedding: List[float]
    object: Literal["embedding"] = "embedding"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: str
    object: Literal["list"] = "list"
    usage: UsageInfo


class OpenAIEmbeddingSpec(LitSpec):
    def __init__(self):
        super().__init__()
        # register the endpoint
        self.add_endpoint("/v1/embeddings", self.embeddings, ["POST"])
        self.add_endpoint("/v1/embeddings", self.options_embeddings, ["GET"])

    def setup(self, server: "LitServer"):  # noqa: F821
        from litserve import LitAPI

        super().setup(server)

        lit_api = self._server.lit_api
        if inspect.isgeneratorfunction(lit_api.predict):
            raise ValueError("OpenAI embedding spec does not support streaming predictions.")

        is_encode_response_original = lit_api.encode_response.__code__ is LitAPI.encode_response.__code__
        if not is_encode_response_original and inspect.isgeneratorfunction(lit_api.encode_response):
            raise ValueError("OpenAI embedding spec does not support streaming predictions.")

        print("OpenAI Embedding Spec is ready.")

    def decode_request(self, request: EmbeddingRequest, context_kwargs: Optional[dict] = None) -> List[str]:
        if isinstance(request.input, str):
            return [request.input]
        return request.input

    def encode_response(self, output: List[List[float]], context_kwargs: Optional[dict] = None) -> dict:
        return {
            "embeddings": output,
            "prompt_tokens": context_kwargs.get("prompt_tokens", 0),
            "total_tokens": context_kwargs.get("total_tokens", 0),
        }

    async def embeddings(self, request: EmbeddingRequest):
        response_queue_id = self.response_queue_id
        logger.debug("Received embedding request: %s", request)
        uid = uuid.uuid4()
        event = asyncio.Event()
        self._server.response_buffer[uid] = event

        self._server.request_queue.put_nowait((response_queue_id, uid, time.monotonic(), request.model_copy()))
        await event.wait()

        response, status = self._server.response_buffer.pop(uid)

        if status == LitAPIStatus.ERROR:
            raise response

        logger.debug(response)

        # TODO: Validate response and also confirm if UsageInfo shoul be default or not, as None is also a valid value

        usage = UsageInfo(**response)
        data = [Embedding(index=i, embedding=embedding) for i, embedding in enumerate(response["embeddings"])]

        return EmbeddingResponse(data=data, model=request.model, usage=usage)

    async def options_embeddings(self, request: Request):
        return Response(status_code=status.HTTP_200_OK)
