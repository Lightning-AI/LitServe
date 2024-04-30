import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

import litserve as ls


@pytest.mark.asyncio()
async def test_simple_pytorch_api():
    api = ls.examples.SimplePyTorchAPI()
    server = ls.LitServer(api)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        response = await ac.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 9.0}


@pytest.mark.asyncio()
async def test_simple_batched_api():
    api = ls.examples.SimpleBatchedAPI()
    server = ls.LitServer(api)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        response = await ac.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}


@pytest.mark.asyncio()
async def test_simple_api():
    api = ls.examples.SimpleLitAPI()
    server = ls.LitServer(api)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        response = await ac.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
