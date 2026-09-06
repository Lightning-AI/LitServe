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
import base64

import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from litserve import LitAPI, LitServer
from litserve.specs import OpenAIEmbeddingSpec
from litserve.utils import wrap_litserve_start


class EncodingAPI(LitAPI):
    def setup(self, device):
        assert device == "cpu"

    def predict(self, inputs):
        texts = [inputs] if isinstance(inputs, str) else inputs
        return [[float(len(text)), -0.5, 0.25] for text in texts]

    def encode_response(self, output):
        return {"embeddings": output, "prompt_tokens": 7, "total_tokens": 7}


@pytest.fixture(scope="module", params=[False, True], ids=["mp", "zmq"])
def embedding_client(request):
    server = LitServer(EncodingAPI(spec=OpenAIEmbeddingSpec()), accelerator="cpu", fast_queue=request.param)
    with wrap_litserve_start(server), TestClient(server.app) as client:
        yield client


@pytest.mark.parametrize("inputs", ["hello", ["hello", "hi"]], ids=["single", "multiple"])
@pytest.mark.parametrize("encoding_format", [None, "float", "base64"], ids=["default", "float", "base64"])
def test_embedding_encoding(embedding_client, inputs, encoding_format):
    payload = {"input": inputs, "model": "local-embedding"}
    if encoding_format is not None:
        payload["encoding_format"] = encoding_format
    response = embedding_client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model"] == "local-embedding"
    assert result["object"] == "list"
    assert result["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
    texts = [inputs] if isinstance(inputs, str) else inputs
    assert len(result["data"]) == len(texts)
    for index, (item, text) in enumerate(zip(result["data"], texts)):
        assert item["index"] == index
        assert item["object"] == "embedding"
        expected = [float(len(text)), -0.5, 0.25]
        if encoding_format == "base64":
            assert isinstance(item["embedding"], str)
            raw = base64.b64decode(item["embedding"], validate=True)
            assert len(raw) == 4 * len(expected)
            assert np.frombuffer(raw, dtype="<f4").tolist() == expected
        else:
            assert item["embedding"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("fast_queue", [False, True], ids=["mp", "zmq"])
async def test_embedding_batch_with_mixed_encodings(fast_queue):
    api = EncodingAPI(spec=OpenAIEmbeddingSpec(), max_batch_size=2, batch_timeout=0.1)
    server = LitServer(api, accelerator="cpu", fast_queue=fast_queue)
    with wrap_litserve_start(server):
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client,
        ):
            responses = await asyncio.gather(
                client.post("/v1/embeddings", json={"input": "hi", "model": "local", "encoding_format": "base64"}),
                client.post("/v1/embeddings", json={"input": "hello", "model": "local", "encoding_format": "float"}),
            )
            for response in responses:
                assert response.status_code == 200, response.text
                assert len(response.json()["data"]) == 1
            encoded = responses[0].json()["data"][0]["embedding"]
            assert isinstance(encoded, str)
            assert np.frombuffer(base64.b64decode(encoded, validate=True), dtype="<f4").tolist() == [2.0, -0.5, 0.25]
            assert responses[1].json()["data"][0]["embedding"] == [5.0, -0.5, 0.25]
