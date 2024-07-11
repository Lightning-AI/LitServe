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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from fastapi import Request, Response
from fastapi.testclient import TestClient
import time

from httpx import AsyncClient

from litserve import LitAPI, LitServer


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_simple():
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=5)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}


def make_load_request(server, outputs):
    with TestClient(server.app) as client:
        for _ in range(100):
            response = client.post("/predict", json={"input": 4.0})
            outputs.append(response.json())


def test_load():
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=25)

    from threading import Thread

    threads = []

    for _ in range(1):
        outputs = []
        t = Thread(target=make_load_request, args=(server, outputs))
        t.start()
        threads.append((t, outputs))

    for t, outputs in threads:
        t.join()
        for el in outputs:
            assert el == {"output": 16.0}


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


@pytest.mark.asyncio()
async def test_timeout():
    api = SlowLitAPI()  # takes 2 second for each prediction
    server = LitServer(api, accelerator="cpu", devices=1, timeout=1.9)

    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        await asyncio.sleep(2)  # Give time to start inference workers
        response1 = ac.post("/predict", json={"input": 4.0})
        response2 = ac.post("/predict", json={"input": 5.0})
        response1, response2 = await asyncio.gather(response1, response2)
        # first request blocks the second request in queue
        # request only times out if it is in queue
        assert response1.status_code == 200, "First request should complete since it's popped from the request queue."
        assert response2.status_code == 504, "Server takes longer than specified timeout and request should timeout"

    # Batched Server
    server = LitServer(SlowBatchAPI(), accelerator="cpu", timeout=1.9, max_batch_size=2, batch_timeout=0.01)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        await asyncio.sleep(2)  # Give time to start inference workers
        response1 = ac.post("/predict", json={"input": 4.0})
        response2 = ac.post("/predict", json={"input": 5.0})
        response3 = ac.post("/predict", json={"input": 6.0})
        response1, response2, response3 = await asyncio.gather(response1, response2, response3)
        assert (
            response1.status_code == 200
        ), "Batch: First request should complete since it's popped from the request queue."
        assert (
            response2.status_code == 200
        ), "Batch: Second request should complete since it's popped from the request queue."

        assert response3.status_code == 504, "Batch: Third request was delayed and should fail"

    server1 = LitServer(SlowLitAPI(), accelerator="cpu", devices=1, timeout=-1)
    server2 = LitServer(SlowLitAPI(), accelerator="cpu", devices=1, timeout=False)
    server3 = LitServer(SlowBatchAPI(), accelerator="cpu", devices=1, timeout=False, max_batch_size=2, batch_timeout=2)
    server4 = LitServer(SlowBatchAPI(), accelerator="cpu", devices=1, timeout=-1, max_batch_size=2, batch_timeout=2)

    with TestClient(server1.app) as client1, TestClient(server2.app) as client2, TestClient(
        server3.app
    ) as client3, TestClient(server4.app) as client4:
        response1 = client1.post("/predict", json={"input": 4.0})
        assert response1.status_code == 200, "Expected slow server to respond since timeout was disabled"

        response2 = client2.post("/predict", json={"input": 4.0})
        assert response2.status_code == 200, "Expected slow server to respond since timeout was disabled"

        response3 = client3.post("/predict", json={"input": 4.0})
        assert response3.status_code == 200, "Expected slow batch server to respond since timeout was disabled"

        response4 = client4.post("/predict", json={"input": 4.0})
        assert response4.status_code == 200, "Expected slow batch server to respond since timeout was disabled"


def test_concurrent_requests():
    n_requests = 100
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)
    with TestClient(server.app) as client, ThreadPoolExecutor(n_requests // 4 + 1) as executor:
        responses = list(executor.map(lambda i: client.post("/predict", json={"input": i}), range(n_requests)))

    count = 0
    for i, response in enumerate(responses):
        assert response.json() == {"output": i**2}, "Server returns square of the input number"
        count += 1
    assert count == n_requests
