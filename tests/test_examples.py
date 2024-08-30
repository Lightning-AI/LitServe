import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from litserve.utils import wrap_litserve_start
import litserve as ls


@pytest.mark.asyncio()
async def test_simple_pytorch_api():
    api = ls.test_examples.SimpleTorchAPI()
    server = ls.LitServer(api, accelerator="cpu")
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            response = await ac.post("/predict", json={"input": 4.0})
            assert response.json() == {"output": 9.0}


@pytest.mark.asyncio()
async def test_simple_batched_api():
    api = ls.test_examples.SimpleBatchedAPI()
    server = ls.LitServer(api, max_batch_size=4, batch_timeout=0.1)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            response = await ac.post("/predict", json={"input": 4.0})
            assert response.json() == {"output": 16.0}


@pytest.mark.asyncio()
async def test_simple_api():
    api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(api)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            response = await ac.post("/predict", json={"input": 4.0})
            assert response.json() == {"output": 16.0}
