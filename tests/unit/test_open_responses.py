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
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.specs.open_responses import OpenResponsesSpec
from litserve.utils import wrap_litserve_start


class TestOpenResponsesAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x, **kwargs):
        # x is CreateResponseRequest
        yield from ["Hello", " ", "World"]


@pytest.fixture
def open_responses_request_data():
    return {
        "input": "Hello",
        "model": "test-model",
    }


@pytest.mark.asyncio
async def test_open_responses_spec_non_streaming(open_responses_request_data):
    api = TestOpenResponsesAPI()
    server = ls.LitServer(api, spec=OpenResponsesSpec())
    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(
                transport=ASGITransport(app=manager.app),
                base_url="http://test",
            ) as ac,
        ):
            open_responses_request_data["stream"] = False
            resp = await ac.post("/v1/responses", json=open_responses_request_data, timeout=10)
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "response"
            assert data["status"] == "completed"
            assert len(data["output"]) == 1
            item = data["output"][0]
            assert item["type"] == "message"
            assert item["role"] == "assistant"
            assert item["content"]["text"] == "Hello World"
