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
from litserve.utils import wrap_litserve_start


class MinimalAsyncAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    async def predict(self, x):
        y = x["input"] ** 2
        return {"output": y}


@pytest.mark.asyncio
async def test_async_api():
    server = ls.LitServer(MinimalAsyncAPI(enable_async=True))
    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            response = await ac.post("/predict", json={"input": 2})
            assert response.json() == {"output": 4}
