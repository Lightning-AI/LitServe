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
from litserve.examples.openai_spec_example import TestAPI, TestAPIWithCustomEncode
from litserve.specs.openai import OpenAISpec, ChatMessage
import litserve as ls


@pytest.mark.asyncio()
async def test_openai_spec(openai_request_data):
    spec = OpenAISpec()
    server = ls.LitServer(TestAPI(), spec=spec)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
        assert resp.status_code == 200, "Status code should be 200"

        assert (
            resp.json()["choices"][0]["message"]["content"] == "This is a generated output"
        ), "LitAPI predict response should match with the generated output"


@pytest.mark.asyncio()
async def test_override_encode(openai_request_data):
    spec = OpenAISpec()
    server = ls.LitServer(TestAPIWithCustomEncode(), spec=spec)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
        assert resp.status_code == 200, "Status code should be 200"

        assert (
            resp.json()["choices"][0]["message"]["content"] == "This is a custom encoded output"
        ), "LitAPI predict response should match with the generated output"


class IncorrectAPI1(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        return "This is a generated output"


class IncorrectAPI2(IncorrectAPI1):
    def predict(self, x):
        yield "This is a generated output"

    def encode_response(self, output):
        return ChatMessage(role="assistant", content="This is a generated output")


@pytest.mark.asyncio()
async def test_openai_spec_validation(openai_request_data):
    spec = OpenAISpec()
    server = ls.LitServer(IncorrectAPI1(), spec=spec)
    with pytest.raises(ValueError, match="predict is not a generator"):
        async with LifespanManager(server.app) as manager:
            await manager.shutdown()

    server = ls.LitServer(IncorrectAPI2(), spec=spec)
    with pytest.raises(ValueError, match="encode_response is not a generator"):
        async with LifespanManager(server.app) as manager:
            await manager.shutdown()
