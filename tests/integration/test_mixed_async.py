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

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.utils import wrap_litserve_start


class MixedStreamingAPI(ls.LitAPI):
    """API with sync predict but async encode_response."""

    def setup(self, device):
        self.model = lambda x: x["input"]

    def predict(self, x):
        # Sync predict
        return self.model(x)

    async def encode_response(self, output):
        # Async encode_response
        await asyncio.sleep(0.01)  # Simulate async I/O
        return {"output": output}


@pytest.mark.asyncio
async def test_mixed_async_integration():
    """Integration test for mixed async/sync API."""
    api = MixedStreamingAPI()
    server = ls.LitServer(api, accelerator="cpu")

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            response = await ac.post("/predict", json={"input": 42})
            assert response.status_code == 200
            assert response.json() == {"output": 42}
