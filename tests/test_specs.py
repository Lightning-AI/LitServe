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

import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from litserve.examples.openai_spec_example import OpenAILitAPI, OpenAISpecWithHooks
from litserve.specs.openai import OpenAISpec
import litserve as ls


@pytest.mark.asyncio()
async def test_openai_spec(openai_request_data):
    spec = OpenAISpec()
    server = ls.LitServer(OpenAILitAPI(), spec=spec)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
        assert resp.status_code == 200, "Status code should be 200"

        assert (
            resp.json()["choices"][0]["message"]["content"] == "encode_response called from LitAPI"
        ), "LitAPI predict response should match with the generated output"


@pytest.mark.asyncio()
async def test_openai_spec_with_hooks(openai_request_data):
    server = ls.LitServer(OpenAILitAPI(), spec=OpenAISpec())
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
        output = response.json()["choices"][0]["message"]["content"]
        expected_output = "encode_response called from LitAPI"

        assert output == expected_output, "Spec didn't have its own encode, and decode so must fallback to LitAPI"

    # Test with hooks, OpenAISpecWithHooks implements decode_request and encode_response
    server = ls.LitServer(OpenAILitAPI(), spec=OpenAISpecWithHooks())
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        response = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
        output = response.json()["choices"][0]["message"]["content"]
        expected_output = "encode_response called from Spec"
        assert output == expected_output, "Spec had its own encode, and decode."
