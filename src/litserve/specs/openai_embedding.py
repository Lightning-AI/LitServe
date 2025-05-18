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
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

from fastapi import HTTPException, Request, Response, status
from fastapi import status as status_code
from pydantic import BaseModel

from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np
    import torch

    from litserve import LitServer


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    dimensions: Optional[int] = None
    encoding_format: Literal["float", "base64"] = "float"
    user: Optional[str] = None

    def get_num_items(self) -> int:
        """Return the number of sentences or tokens in the input."""
        if isinstance(self.input, list):
            if isinstance(self.input[0], list):
                return len(self.input[0])
            return len(self.input)
        return 1

    def ensure_list(self):
        return self.input if isinstance(self.input, list) else [self.input]


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


EMBEDDING_API_EXAMPLE = """
Please follow the example below for guidance on how to use the OpenAI Embedding spec:

```python
import numpy as np
from typing import List
from litserve.specs import EmbeddingRequest
from litserve import LitAPI, OpenAIEmbeddingSpec

class TestAPI(LitAPI):
    def setup(self, device):
        self.model = None

    def decode_request(self, request: EmbeddingRequest) -> List[str]:
        return request.input

    def predict(self, x) -> List[List[float]]:
        return np.random.rand(len(x), 768).tolist()

    def encode_response(self, output) -> dict:
        return {"embeddings": output}

if __name__ == "__main__":
    import litserve as ls
    server = ls.LitServer(TestAPI(), spec=OpenAIEmbeddingSpec())
    server.run()
```
"""


class OpenAIEmbeddingSpec(LitSpec):
    def __init__(self):
        super().__init__()
        # register the endpoint
        self.add_endpoint("/v1/embeddings", self.embeddings_endpoint, ["POST"])
        self.add_endpoint("/v1/embeddings", self.options_embeddings, ["GET"])

    def setup(self, server: "LitServer"):
        from litserve import LitAPI

        super().setup(server)

        lit_api = self._server.lit_api
        if inspect.isgeneratorfunction(lit_api.predict):
            raise ValueError(
                "You are using yield in your predict method, which is used for streaming.",
                "OpenAIEmbeddingSpec doesn't support streaming because producing embeddings ",
                "is not a sequential operation.",
                "Please consider replacing yield with return in predict.\n",
                EMBEDDING_API_EXAMPLE,
            )

        is_encode_response_original = lit_api.encode_response.__code__ is LitAPI.encode_response.__code__
        if not is_encode_response_original and inspect.isgeneratorfunction(lit_api.encode_response):
            raise ValueError(
                "You are using yield in your encode_response method, which is used for streaming.",
                "OpenAIEmbeddingSpec doesn't support streaming because producing embeddings ",
                "is not a sequential operation.",
                "Please consider replacing yield with return in encode_response.\n",
                EMBEDDING_API_EXAMPLE,
            )

        print("OpenAI Embedding Spec is ready.")

    def decode_request(self, request: EmbeddingRequest, context_kwargs: Optional[dict] = None) -> List[str]:
        return request.input

    def encode_response(
        self, output: List[List[float]], context_kwargs: Optional[dict] = None
    ) -> Union[dict, EmbeddingResponse]:
        usage = {
            "prompt_tokens": context_kwargs.get("prompt_tokens", 0) if context_kwargs else 0,
            "total_tokens": context_kwargs.get("total_tokens", 0) if context_kwargs else 0,
        }
        return {"embeddings": output} | usage

    def _validate_response(self, response: Union[dict, List[Embedding], Any]) -> None:
        if isinstance(response, List[Embedding]):
            return
        if not isinstance(response, (dict, EmbeddingResponse)):
            raise ValueError(
                f"Expected response to be a dictionary, but got type {type(response)}.",
                "The response should be a dictionary to ensure proper compatibility with the OpenAIEmbeddingSpec.\n\n"
                "Please ensure that your response is a dictionary with the following keys:\n"
                "- 'embeddings' (required)\n"
                "- 'prompt_tokens' (optional)\n"
                "- 'total_tokens' (optional)\n"
                f"{EMBEDDING_API_EXAMPLE}",
            )
        if "embeddings" not in response:
            raise ValueError(
                "The response does not contain the key 'embeddings'."
                "The key 'embeddings' is required to ensure proper compatibility with the OpenAIEmbeddingSpec.\n"
                "Please ensure that your response contains the key 'embeddings'.\n"
                f"{EMBEDDING_API_EXAMPLE}"
            )

    def _handle_embedding_response(
        self, embeddings: Union[List, "np.ndarray", "torch.Tensor", "List[List[float]]"], num_items: int = 1
    ) -> List[Embedding]:
        ndim = None
        if "torch" in sys.modules:
            import torch

            if isinstance(embeddings, torch.Tensor):
                ndim = embeddings.ndim
        if "numpy" in sys.modules:
            import numpy as np

            if isinstance(embeddings, np.ndarray):
                ndim = embeddings.ndim

        if ndim == 1:
            embeddings = embeddings[None, :]

        if ndim is not None:
            embeddings = embeddings.tolist()

        # check if we have total num_items number of embeddings vectors
        num_response_items = len(embeddings)
        if num_response_items != num_items:
            raise ValueError(f"Expected {num_items} embeddings, but got {len(embeddings)}")

        result = []
        for i, embedding in enumerate(embeddings):
            result.append(Embedding(index=i, embedding=embedding))

        return result

    async def embeddings_endpoint(self, request: EmbeddingRequest) -> EmbeddingResponse:
        response_queue_id = self.response_queue_id
        logger.debug("Received embedding request: %s", request)
        uid = uuid.uuid4()
        event = asyncio.Event()
        self._server.response_buffer[uid] = event

        self._server.request_queue.put_nowait((response_queue_id, uid, time.monotonic(), request.model_copy()))
        await event.wait()

        response, status = self._server.response_buffer.pop(uid)

        if status == LitAPIStatus.ERROR and isinstance(response, HTTPException):
            logger.error("Error in embedding request: %s", response)
            raise response
        if status == LitAPIStatus.ERROR:
            logger.error("Error in embedding request: %s", response)
            raise HTTPException(status_code=status_code.HTTP_500_INTERNAL_SERVER_ERROR)

        logger.debug(response)

        self._validate_response(response)
        data: List[Embedding] = self._handle_embedding_response(response["embeddings"], request.get_num_items())

        usage = UsageInfo(**response)

        return EmbeddingResponse(data=data, model=request.model, usage=usage)

    async def options_embeddings(self, request: Request):
        return Response(status_code=status.HTTP_200_OK)
