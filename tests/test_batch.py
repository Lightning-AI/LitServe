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
import time
import warnings
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from asgi_lifespan import LifespanManager
from fastapi import Request, Response
from httpx import AsyncClient
from litserve import LitAPI, LitServer
from litserve.server import run_batched_loop


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.fill_(2.0)
        self.linear.bias.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request):
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def batch(self, inputs):
        assert len(inputs) == 2
        return torch.stack(inputs)

    def predict(self, x):
        assert len(x) == 2, "Expected two concurrent inputs to be batched"
        return self.model(x)

    def unbatch(self, output):
        assert output.shape[0] == 2
        return list(output)

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


class SimpleLitAPI2(LitAPI):
    def setup(self, device):
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request):
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def predict(self, x):
        assert x.shape == (1,), f"{x}"
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


@pytest.mark.asyncio()
async def test_batched():
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cpu", devices=1, timeout=10, max_batch_size=2, batch_timeout=4)

    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        response1 = ac.post("/predict", json={"input": 4.0})
        response2 = ac.post("/predict", json={"input": 5.0})
        response1, response2 = await asyncio.gather(response1, response2)

    assert response1.json() == {"output": 9.0}
    assert response2.json() == {"output": 11.0}


@pytest.mark.asyncio()
async def test_unbatched():
    api = SimpleLitAPI2()
    server = LitServer(api, accelerator="cpu", devices=1, timeout=10, max_batch_size=1)

    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        response1 = ac.post("/predict", json={"input": 4.0})
        response2 = ac.post("/predict", json={"input": 5.0})
        response1, response2 = await asyncio.gather(response1, response2)

    assert response1.json() == {"output": 9.0}
    assert response2.json() == {"output": 11.0}


def test_max_batch_size():
    with pytest.raises(ValueError, match="must be"):
        LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=2, max_batch_size=0)

    with pytest.raises(ValueError, match="must be"):
        LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=2, max_batch_size=-1)

    with pytest.raises(ValueError, match="must be"):
        LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=2, max_batch_size=2, batch_timeout=5)


def test_max_batch_size_warning():
    warning = "both batch and unbatch methods implemented, but the max_batch_size parameter was not set."
    with pytest.warns(
        UserWarning,
        match=warning,
    ):
        LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=2)

    # Test no warnings are raised when max_batch_size is set
    with pytest.raises(pytest.fail.Exception), pytest.warns(
        UserWarning,
        match=warning,
    ):
        LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=2, max_batch_size=2)

    # Test no max_batch_size warnings are raised with a different API
    with pytest.raises(pytest.fail.Exception), pytest.warns(
        UserWarning,
        match=warning,
    ):
        LitServer(SimpleLitAPI2(), accelerator="cpu", devices=1, timeout=2)


class FakeResponseQueue:
    def put(self, *args):
        raise Exception("Exit loop")


def test_batched_loop():
    requests_queue = Queue()
    requests_queue.put(("uuid-1234", time.monotonic(), {"input": 4.0}))
    requests_queue.put(("uuid-1235", time.monotonic(), {"input": 5.0}))

    lit_api_mock = MagicMock()
    lit_api_mock.request_timeout = 2
    lit_api_mock.decode_request = MagicMock(side_effect=lambda x: x["input"])
    lit_api_mock.batch = MagicMock(side_effect=lambda x: x)
    lit_api_mock.predict = MagicMock(side_effect=lambda x: [16.0, 25.0])
    lit_api_mock.unbatch = MagicMock(side_effect=lambda x: x)
    lit_api_mock.encode_response = MagicMock(side_effect=lambda x: {"output": x})

    with patch("pickle.dumps", side_effect=StopIteration("exit loop")), pytest.raises(StopIteration, match="exit loop"):
        run_batched_loop(
            lit_api_mock, lit_api_mock, requests_queue, FakeResponseQueue(), max_batch_size=2, batch_timeout=4
        )

    lit_api_mock.batch.assert_called_once()
    lit_api_mock.batch.assert_called_once_with([4.0, 5.0])
    lit_api_mock.unbatch.assert_called_once()
