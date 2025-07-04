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
import copy
import time

import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.specs.openai import OpenAISpec
from litserve.specs.openai_embedding import OpenAIEmbeddingSpec
from litserve.test_examples.openai_embedding_spec_example import (
    TestEmbedAPI,
    TestOpenAPI,
    TestEmbedAPIWithMissingEmbeddings,
    TestEmbedAPIWithNonDictOutput,
    TestEmbedAPIWithUsage,
    TestEmbedAPIWithYieldEncodeResponse,
    TestEmbedAPIWithYieldPredict,
)
from litserve.utils import wrap_litserve_start


@pytest.mark.asyncio
async def test_openai_embedding_spec_with_single_input(openai_embedding_request_data):
    spec = OpenAIEmbeddingSpec()
    server = ls.LitServer(TestEmbedAPI(spec=spec))

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/embeddings", json=openai_embedding_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            assert resp.json()["object"] == "list", "Object should be list"
            assert resp.json()["data"][0]["index"] == 0, "Index should be 0"
            assert len(resp.json()["data"]) == 1, "Length of data should be 1"
            assert len(resp.json()["data"][0]["embedding"]) == 768, "Embedding length should be 768"

@pytest.mark.asyncio
async def test_openai_embedding_spec_with_multi_endpoint(openai_embedding_request_data):
    spec_openai = OpenAISpec()
    spec_embedding = OpenAIEmbeddingSpec()
    server = ls.LitServer([TestOpenAPI(spec=spec_openai,enable_async=True),TestEmbedAPI(spec=spec_embedding)])
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/embeddings", json=openai_embedding_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            assert resp.json()["object"] == "list", "Object should be list"
            assert resp.json()["data"][0]["index"] == 0, "Index should be 0"
            assert len(resp.json()["data"]) == 1, "Length of data should be 1"
            assert len(resp.json()["data"][0]["embedding"]) == 768, "Embedding length should be 768"


@pytest.mark.asyncio
async def test_openai_embedding_spec_with_multiple_inputs(openai_embedding_request_data_array):
    spec = OpenAIEmbeddingSpec()
    server = ls.LitServer(TestEmbedAPI(spec=spec))
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/embeddings", json=openai_embedding_request_data_array, timeout=10)
            assert resp.status_code == 200, (
                f"Status code should be 200 but got {resp.status_code}, response: {resp.content}"
            )
            assert resp.json()["object"] == "list", "Object should be list"
            assert resp.json()["data"][0]["index"] == 0, "Index should be 0"
            assert len(resp.json()["data"]) == 4, "Length of data should be 1"
            assert len(resp.json()["data"][0]["embedding"]) == 768, "Embedding length should be 768"


@pytest.mark.asyncio
async def test_openai_embedding_spec_with_usage(openai_embedding_request_data):
    spec = OpenAIEmbeddingSpec()
    server = ls.LitServer(TestEmbedAPIWithUsage(spec=spec))

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/embeddings", json=openai_embedding_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            assert resp.json()["object"] == "list", "Object should be list"
            assert resp.json()["data"][0]["index"] == 0, "Index should be 0"
            assert len(resp.json()["data"]) == 1, "Length of data should be 1"
            assert len(resp.json()["data"][0]["embedding"]) == 768, "Embedding length should be 768"
            assert resp.json()["usage"]["prompt_tokens"] == 10, "Prompt tokens should be 10"
            assert resp.json()["usage"]["total_tokens"] == 10, "Total tokens should be 10"


@pytest.mark.asyncio
async def test_openai_embedding_spec_validation(openai_request_data):
    server = ls.LitServer(TestEmbedAPIWithYieldPredict(), spec=OpenAIEmbeddingSpec())
    with pytest.raises(ValueError, match="You are using yield in your predict method"), wrap_litserve_start(
        server
    ) as server:
        async with LifespanManager(server.app):
            pass

    server = ls.LitServer(TestEmbedAPIWithYieldEncodeResponse(), spec=OpenAIEmbeddingSpec())
    with pytest.raises(ValueError, match="You are using yield in your encode_response method"), wrap_litserve_start(
        server
    ) as server:
        async with LifespanManager(server.app):
            pass


@pytest.mark.asyncio
async def test_openai_embedding_spec_with_non_dict_output(openai_embedding_request_data):
    server = ls.LitServer(TestEmbedAPIWithNonDictOutput(), spec=ls.OpenAIEmbeddingSpec())

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            with pytest.raises(ValueError, match="Expected response to be a dictionary"):
                await ac.post("/v1/embeddings", json=openai_embedding_request_data, timeout=10)


@pytest.mark.asyncio
async def test_openai_embedding_spec_with_missing_embeddings(openai_embedding_request_data):
    server = ls.LitServer(TestEmbedAPIWithMissingEmbeddings(), spec=OpenAIEmbeddingSpec())

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            with pytest.raises(ValueError, match="The response does not contain the key 'embeddings'"):
                await ac.post("/v1/embeddings", json=openai_embedding_request_data, timeout=10)


class TestOpenAIWithBatching(TestEmbedAPI):
    def predict(self, batch):
        time.sleep(2)
        return np.random.rand(len(batch), 768).tolist()


@pytest.mark.asyncio
async def test_openai_embedding_spec_with_batching(openai_embedding_request_data):
    server = ls.LitServer(TestOpenAIWithBatching(max_batch_size=10, batch_timeout=4), spec=ls.OpenAIEmbeddingSpec())

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # send concurrent requests
            req1 = copy.deepcopy(openai_embedding_request_data)
            req2 = copy.deepcopy(openai_embedding_request_data)
            req2["input"] = "This is the second request"
            tasks = []
            t0 = time.perf_counter()
            for _ in range(5):
                tasks.append(ac.post("/v1/embeddings", json=req1, timeout=10))
                tasks.append(ac.post("/v1/embeddings", json=req2, timeout=10))

            responses = await asyncio.gather(*tasks)
            t1 = time.perf_counter()
            print(f"Time taken: {t1 - t0} seconds")
            for resp in responses:
                assert resp.status_code == 200, (
                    f"Status code should be 200, but got {resp.status_code}, response: {resp.content}"
                )
                assert len(resp.json()["data"]) == 1, "Length of data should be 1"
            assert t1 - t0 < 20, "Time taken must be less than 20 seconds (batching is not working)"


@pytest.mark.asyncio
async def test_batching_with_client_side_batching(openai_embedding_request_data_array):
    server = ls.LitServer(TestOpenAIWithBatching(max_batch_size=2, batch_timeout=10), spec=ls.OpenAIEmbeddingSpec())

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/embeddings", json=openai_embedding_request_data_array, timeout=10)

            assert resp.status_code == 400, "Cient side batching is not supported with dynamic batching"
            assert (
                resp.json()["detail"]
                == "The OpenAIEmbedding spec does not support dynamic batching when client-side batching is used. "
                "To resolve this, either set `max_batch_size=1` or send a single input from the client."
            )
