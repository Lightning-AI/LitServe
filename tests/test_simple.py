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
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from fastapi import Request, Response, status
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import litserve.server
from litserve import LitAPI, LitServer
from litserve.utils import wrap_litserve_start


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_simple(lit_server):
    with TestClient(lit_server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}


class SlowSetupLitAPI(SimpleLitAPI):
    def setup(self, device):
        self.model = lambda x: x**2
        time.sleep(2)


class SlowSetupWithCustomHealthLitAPI(SimpleLitAPI):
    def setup(self, device):
        self.model = lambda x: x**2
        time.sleep(2)

    def health(self) -> bool:
        # Health check passes after 5 seconds from the first time it is called.
        if not hasattr(self, "_start_time"):
            self._start_time = time.time()
        return time.time() - self._start_time >= 5


@pytest.mark.parametrize("use_zmq", [True, False])
def test_workers_health(use_zmq):
    server = LitServer(
        SlowSetupLitAPI(), accelerator="cpu", devices=1, timeout=5, workers_per_device=2, fast_queue=use_zmq
    )

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.text == "not ready"

        time.sleep(1)
        response = client.get("/health")
        assert response.status_code == 503
        assert response.text == "not ready"

        time.sleep(3)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.text == "ok"


@pytest.mark.parametrize("use_zmq", [True, False])
def test_workers_health_custom_path(use_zmq):
    server = LitServer(
        SlowSetupLitAPI(),
        accelerator="cpu",
        healthcheck_path="/my_server/health",
        devices=1,
        timeout=5,
        workers_per_device=2,
        fast_queue=use_zmq,
    )

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.get("/my_server/health")
        assert response.status_code == 503
        assert response.text == "not ready"

        time.sleep(1)
        response = client.get("/my_server/health")
        assert response.status_code == 503
        assert response.text == "not ready"

        time.sleep(3)
        response = client.get("/my_server/health")
        assert response.status_code == 200
        assert response.text == "ok"


@pytest.mark.parametrize("use_zmq", [True, False])
def test_workers_health_with_custom_health_method(use_zmq):
    server = LitServer(
        SlowSetupWithCustomHealthLitAPI(),
        accelerator="cpu",
        devices=1,
        timeout=5,
        workers_per_device=2,
        fast_queue=use_zmq,
    )

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.text == "not ready"

        time.sleep(1)
        response = client.get("/health")
        assert response.status_code == 503
        assert response.text == "not ready"

        time.sleep(1)
        response = client.get("/health")
        assert response.status_code == 503
        assert response.text == "not ready"

        time.sleep(4)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.text == "ok"


def make_load_request(server, outputs):
    with TestClient(server.app) as client:
        for i in range(100):
            response = client.post("/predict", json={"input": i})
            outputs.append(response.json())


def test_load(lit_server):
    from threading import Thread

    threads = []
    for _ in range(1):
        outputs = []
        t = Thread(target=make_load_request, args=(lit_server, outputs))
        t.start()
        threads.append((t, outputs))

    for t, outputs in threads:
        t.join()
        for i, el in enumerate(outputs):
            assert el == {"output": i**2}


@pytest.fixture
def shutdown_api_key():
    api_key = f"test-key-{random.randint(100, 1000000)}"
    litserve.server.SHUTDOWN_API_KEY = api_key
    yield api_key
    litserve.server.SHUTDOWN_API_KEY = None


def test_shutdown_endpoint_single_worker(shutdown_api_key):
    server = LitServer(
        SimpleLitAPI(),
        accelerator="cpu",
        devices=1,
        workers_per_device=1,
        enable_shutdown_api=True,
    )

    with wrap_litserve_start(server) as srv, TestClient(srv.app) as client:
        response_no_header = client.post("/shutdown")
        assert response_no_header.status_code == 401

        response_correct_key = client.post("/shutdown", headers={"Authorization": f"Bearer {shutdown_api_key}"})
        assert response_correct_key.status_code == status.HTTP_200_OK


def test_shutdown_endpoint_multiple_workers(shutdown_api_key):
    server = LitServer(
        SimpleLitAPI(),
        accelerator="cpu",
        devices=1,
        workers_per_device=3,
        enable_shutdown_api=True,
    )

    with wrap_litserve_start(server) as srv, TestClient(srv.app) as client:
        response_no_header = client.post("/shutdown")
        assert response_no_header.status_code == 401

        response_wrong_key = client.post("/shutdown", headers={"Authorization": "Bearer wrong_key"})
        assert response_wrong_key.status_code == 401

        response_correct_key = client.post("/shutdown", headers={"Authorization": f"Bearer {shutdown_api_key}"})
        assert response_correct_key.status_code == status.HTTP_200_OK


class SlowLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        time.sleep(2)
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


class SlowBatchAPI(SlowLitAPI):
    def batch(self, inputs):
        return np.asarray(inputs)

    def unbatch(self, output):
        return list(output)


@pytest.mark.flaky(retries=3)
@pytest.mark.parametrize("use_zmq", [True, False])
@pytest.mark.asyncio
async def test_timeout(use_zmq):
    # Scenario: first request completes, second request times out in queue
    api = SlowLitAPI()  # takes 2 seconds for each prediction
    server = LitServer(api, accelerator="cpu", devices=1, timeout=2, fast_queue=use_zmq)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            response1 = asyncio.create_task(ac.post("/predict", json={"input": 4.0}))
            await asyncio.sleep(0.0001)
            response2 = asyncio.create_task(ac.post("/predict", json={"input": 5.0}))
            responses = await asyncio.gather(response1, response2, return_exceptions=True)
            assert responses[0].status_code == 200, (
                "First request should complete since it's popped from the request queue."
            )
            assert responses[1].status_code == 504, (
                "Server takes longer than specified timeout and request should timeout"
            )


@pytest.mark.flaky(retries=3)
@pytest.mark.parametrize("use_zmq", [True, False])
@pytest.mark.asyncio
async def test_batch_timeout_with_concurrent_requests(use_zmq):
    server = LitServer(
        SlowBatchAPI(
            max_batch_size=2,
            batch_timeout=0.01,
        ),
        accelerator="cpu",
        timeout=2,
        fast_queue=use_zmq,
    )
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            response1 = asyncio.create_task(ac.post("/predict", json={"input": 4.0}))
            response2 = asyncio.create_task(ac.post("/predict", json={"input": 5.0}))
            await asyncio.sleep(0.0001)
            response3 = asyncio.create_task(ac.post("/predict", json={"input": 6.0}))
            responses = await asyncio.gather(response1, response2, response3, return_exceptions=True)

            assert responses[0].status_code == 200, "First request in batch should complete"
            assert responses[1].status_code == 200, "Second request in batch should complete"
            assert responses[2].status_code == 504, "Third request should timeout"


@pytest.mark.parametrize("use_zmq", [True, False])
def test_server_with_disabled_timeout(use_zmq):
    servers = [
        LitServer(SlowLitAPI(), accelerator="cpu", devices=1, timeout=-1),
        LitServer(SlowLitAPI(), accelerator="cpu", devices=1, timeout=False),
        LitServer(
            SlowBatchAPI(
                max_batch_size=2,
                batch_timeout=2,
            ),
            accelerator="cpu",
            devices=1,
            timeout=False,
            fast_queue=use_zmq,
        ),
        LitServer(
            SlowBatchAPI(
                max_batch_size=2,
                batch_timeout=2,
            ),
            accelerator="cpu",
            devices=1,
            timeout=-1,
            fast_queue=use_zmq,
        ),
    ]

    with ExitStack() as stack:
        clients = [
            stack.enter_context(TestClient(stack.enter_context(wrap_litserve_start(server)).app)) for server in servers
        ]

        for i, client in enumerate(clients, 1):
            response = client.post("/predict", json={"input": 4.0})
            assert response.status_code == 200, f"Server {i} should complete request with disabled timeout"


def test_concurrent_requests(lit_server):
    n_requests = 100
    with TestClient(lit_server.app) as client, ThreadPoolExecutor(n_requests // 4 + 1) as executor:
        responses = list(executor.map(lambda i: client.post("/predict", json={"input": i}), range(n_requests)))

    count = 0
    for i, response in enumerate(responses):
        assert response.json() == {"output": i**2}, "Server returns square of the input number"
        count += 1
    assert count == n_requests


class CustomError(Exception):
    def __init__(self, arg1, arg2, arg3):
        super().__init__("Test exception")


class ExceptionAPI(SimpleLitAPI):
    def predict(self, x):
        raise CustomError("This", "is", "a test")


def test_exception():
    server = LitServer(ExceptionAPI(), accelerator="cpu", devices=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
