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
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            response = await ac.post("/predict", json={"input": 2})
            assert response.json() == {"output": 4}
