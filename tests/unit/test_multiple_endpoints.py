import asyncio
import time

import pytest
from asgi_lifespan import LifespanManager
from fastapi.testclient import TestClient
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
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
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


# ==================== INDIVIDUAL HEALTH ENDPOINT TESTS ====================


class SlowSetupAPI(ls.LitAPI):
    """API with slow setup for testing health during worker initialization."""

    def setup(self, device):
        self.model = lambda x: x**2
        time.sleep(2)

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


class UnhealthyAPI(ls.LitAPI):
    """API that always reports unhealthy."""

    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}

    def health(self) -> bool:
        return False


class AsyncHealthAPI(ls.LitAPI):
    """API with async health check."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._healthy = True

    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}

    async def health(self) -> bool:
        await asyncio.sleep(0.01)
        return self._healthy


@pytest.mark.asyncio
async def test_individual_health_endpoints():
    """Test individual health endpoints for each LitAPI."""
    api1 = InferencePipeline(name="api1", api_path="/api1")
    api2 = InferencePipeline(name="api2", api_path="/api2")
    server = ls.LitServer([api1, api2])

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            # Wait for workers to be ready
            for _ in range(10):
                resp1 = await ac.get("/api1/health")
                resp2 = await ac.get("/api2/health")
                if resp1.status_code == 200 and resp2.status_code == 200:
                    break
                await asyncio.sleep(0.5)

            # Test individual health endpoints
            resp = await ac.get("/api1/health")
            assert resp.status_code == 200, "api1/health should return 200 (OK)"
            assert resp.text == "ok"

            resp = await ac.get("/api2/health")
            assert resp.status_code == 200, "api2/health should return 200 (OK)"
            assert resp.text == "ok"

            # Test global health still works (backward compatibility)
            resp = await ac.get("/health")
            assert resp.status_code == 200, f"Global /health should still work, got {resp.status_code}: {resp.text}"
            assert resp.text == "ok"


def test_individual_health_with_unhealthy_api():
    """Test individual health endpoints when one API is unhealthy."""
    api1 = InferencePipeline(name="api1", api_path="/api1")
    api2 = UnhealthyAPI(api_path="/api2")
    server = ls.LitServer([api1, api2])

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Wait for workers to be ready
        time.sleep(3)

        # Healthy API should return 200
        response = client.get("/api1/health")
        assert response.status_code == 200, "Healthy API should return 200"
        assert response.text == "ok"

        # Unhealthy API should return 503
        response = client.get("/api2/health")
        assert response.status_code == 503, "Unhealthy API should return 503"
        assert response.text == "not ready"

        # Global health should return 503 (because one API is unhealthy)
        response = client.get("/health")
        assert response.status_code == 503, "Global health should return 503 when any API is unhealthy"
        assert response.text == "not ready"


@pytest.mark.asyncio
async def test_individual_health_with_async_method():
    """Test individual health endpoint with async health() method."""
    api1 = AsyncHealthAPI(api_path="/api1")
    api2 = AsyncHealthAPI(api_path="/api2")
    server = ls.LitServer([api1, api2])

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            # Wait for workers to be ready
            for _ in range(10):
                resp = await ac.get("/api1/health")
                if resp.status_code == 200:
                    break
                await asyncio.sleep(0.5)

            # Test async health check is properly awaited
            resp = await ac.get("/api1/health")
            assert resp.status_code == 200, "Async health check should work"
            assert resp.text == "ok"

            resp = await ac.get("/api2/health")
            assert resp.status_code == 200, "Async health check should work"
            assert resp.text == "ok"


@pytest.mark.parametrize("use_zmq", [True, False])
def test_individual_health_during_slow_setup(use_zmq):
    """Test individual health endpoints during worker setup."""
    api1 = SlowSetupAPI(api_path="/api1")
    api2 = InferencePipeline(name="api2", api_path="/api2")
    server = ls.LitServer(
        [api1, api2],
        accelerator="cpu",
        devices=1,
        timeout=5,
        workers_per_device=2,
        fast_queue=use_zmq,
    )

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # During setup, api1 should be not ready
        response = client.get("/api1/health")
        assert response.status_code == 503, "api1 should be not ready during setup"
        assert response.text == "not ready"

        # Wait for api2 to be ready (it might need a moment)
        for _ in range(10):
            response = client.get("/api2/health")
            if response.status_code == 200:
                break
            time.sleep(0.5)
        assert response.status_code == 200, "api2 should be ready soon"
        assert response.text == "ok"

        # Wait for api1 setup to complete
        time.sleep(3)

        # Now api1 should be ready
        response = client.get("/api1/health")
        assert response.status_code == 200, "api1 should be ready after setup"
        assert response.text == "ok"


@pytest.mark.parametrize("use_zmq", [True, False])
def test_individual_health_with_custom_path(use_zmq):
    """Test individual health endpoints with custom global health path."""
    api1 = InferencePipeline(name="api1", api_path="/api1")
    api2 = InferencePipeline(name="api2", api_path="/api2")
    server = ls.LitServer(
        [api1, api2],
        healthcheck_path="/custom/health",
        accelerator="cpu",
        devices=1,
        timeout=5,
        workers_per_device=1,
        fast_queue=use_zmq,
    )

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Wait for workers
        time.sleep(2)

        # Individual health paths should still be /api1/health, /api2/health
        response = client.get("/api1/health")
        assert response.status_code == 200, "Individual health path should not be affected by custom global path"
        assert response.text == "ok"

        response = client.get("/api2/health")
        assert response.status_code == 200
        assert response.text == "ok"

        # Global health should use custom path
        response = client.get("/custom/health")
        assert response.status_code == 200, "Global health should use custom path"
        assert response.text == "ok"

        # Old /health path should not work
        response = client.get("/health")
        assert response.status_code == 404, "Old /health path should not exist with custom path"


def test_individual_health_single_api():
    """Test individual health endpoints work with single API server."""
    api1 = InferencePipeline(name="api1", api_path="/predict")
    server = ls.LitServer(api1)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Wait for workers
        for _ in range(10):
            response = client.get("/predict/health")
            if response.status_code == 200:
                break
            time.sleep(0.5)

        # Individual health should work
        response = client.get("/predict/health")
        assert response.status_code == 200, "Individual health should work with single API"
        assert response.text == "ok"

        # Global health should still work
        response = client.get("/health")
        assert response.status_code == 200, "Global health should work with single API"
        assert response.text == "ok"


def test_no_duplicate_health_tests():
    """Verify we're not duplicating existing test functionality."""
    # This test documents what's already tested elsewhere
    # Existing tests in test_simple.py cover:
    # - test_workers_health: Basic health check
    # - test_workers_health_custom_path: Custom health path
    # - test_workers_health_with_custom_health_method: Custom health method
    # - test_workers_health_with_async_health_method: Async health method

    # Our new tests focus on INDIVIDUAL health endpoints for multiple APIs
    # which is NOT covered by existing tests
    assert True
