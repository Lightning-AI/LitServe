# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import asyncio
from unittest.mock import MagicMock
from asgi_lifespan import LifespanManager

from litserve.server import run_batched_loop

import pytest

from fastapi import Request, Response
from httpx import AsyncClient

from litserve import LitAPI, LitServer

import torch
import torch.nn as nn


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
        assert len(x) == 2
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


@pytest.mark.asyncio
async def test_batched():
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cpu", devices=1, timeout=10, max_batch_size=2, batch_timeout=4)

    async with LifespanManager(server.app) as manager:
        async with AsyncClient(app=manager.app, base_url="http://test") as ac:
            response1 = ac.post("/predict", json={"input": 4.0})
            response2 = ac.post("/predict", json={"input": 5.0})
            response1, response2 = await asyncio.gather(response1, response2)

    assert response1.json() == {"output": 9.0}
    assert response2.json() == {"output": 11.0}


@pytest.mark.asyncio
async def test_unbatched():
    api = SimpleLitAPI2()
    server = LitServer(api, accelerator="cpu", devices=1, timeout=10, max_batch_size=1)

    async with LifespanManager(server.app) as manager:
        async with AsyncClient(app=manager.app, base_url="http://test") as ac:
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


def test_batched_loop():
    from multiprocessing import Manager, Queue, Pipe

    requests_queue = Queue()
    request_buffer = Manager().dict()
    requests_queue.put(1)
    requests_queue.put(2)
    read, write = Pipe()
    request_buffer[1] = {"input": 4.0}, write
    request_buffer[2] = {"input": 5.0}, write

    lit_api_mock = MagicMock()
    lit_api_mock.decode_request = MagicMock(side_effect=lambda x: x["input"])
    lit_api_mock.batch = MagicMock()
    lit_api_mock.unbatch = MagicMock(side_effect=Exception("exit loop"))

    with pytest.raises(Exception, match="exit loop"):
        run_batched_loop(lit_api_mock, requests_queue, request_buffer, max_batch_size=2, batch_timeout=4)

    lit_api_mock.batch.assert_called_once()
    lit_api_mock.batch.assert_called_once_with((4.0, 5.0))
    lit_api_mock.unbatch.assert_called_once()
