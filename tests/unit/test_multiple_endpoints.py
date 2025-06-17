import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.utils import wrap_litserve_start


class InferencePipeline(ls.LitAPI):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output, "name": self.name}


@pytest.mark.asyncio
async def test_multiple_endpoints():
    api1 = InferencePipeline(name="api1", api_path="/api1")
    api2 = InferencePipeline(name="api2", api_path="/api2")
    server = ls.LitServer([api1, api2])

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/api1", json={"input": 2.0}, timeout=10)
            assert resp.status_code == 200, "Server response should be 200 (OK)"
            assert resp.json()["output"] == 4.0, "output from Identity server must be same as input"
            assert resp.json()["name"] == "api1", "name from Identity server must be same as input"

            resp = await ac.post("/api2", json={"input": 5.0}, timeout=10)
            assert resp.status_code == 200, "Server response should be 200 (OK)"
            assert resp.json()["output"] == 25.0, "output from Identity server must be same as input"
            assert resp.json()["name"] == "api2", "name from Identity server must be same as input"


def test_multiple_endpoints_with_same_path():
    api1 = InferencePipeline(name="api1", api_path="/api1")
    api2 = InferencePipeline(name="api2", api_path="/api1")
    with pytest.raises(ValueError, match="api_path /api1 is already in use by"):
        ls.LitServer([api1, api2])


def test_reserved_paths():
    api1 = InferencePipeline(name="api1", api_path="/health")
    api2 = InferencePipeline(name="api2", api_path="/info")
    with pytest.raises(ValueError, match="api_path /health is already in use by LitServe healthcheck"):
        ls.LitServer([api1, api2])
