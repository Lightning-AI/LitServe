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
from concurrent.futures import ThreadPoolExecutor

from fastapi import Request, Response
from fastapi.testclient import TestClient
import time


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
        time.sleep(1)
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_timeout():
    server = LitServer(SlowLitAPI(), accelerator="cpu", devices=1, timeout=0.5)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 504


def test_concurrent_requests():
    n_requests = 100
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)
    with TestClient(server.app) as client, ThreadPoolExecutor(n_requests // 4 + 1) as executor:
        responses = list(executor.map(lambda _: client.post("/predict", json={"input": 4.0}), range(n_requests)))

    count = 0
    for response in responses:
        assert response.json() == {"output": 16.0}
        count += 1
    assert count == n_requests
