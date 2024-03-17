# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from fastapi import Request, Response
# from fastapi.testclient import TestClient
from httpx import AsyncClient

from litserve import LitAPI, LitServer
import litserve.server

import torch
import torch.nn as nn

import pytest


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
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


def patched_inference_worker(*args, **kwargs):
    # TODO: delay the start of the loop, so requests have time to accumulate on the queue
    import time
    time.sleep(0.5)
    litserve.server.inference_worker_(*args, **kwargs)


@pytest.mark.anyio
async def test_batched(monkeypatch):
    monkeypatch.setattr(litserve.server, "inference_worker_", litserve.server.inference_worker)
    monkeypatch.setattr(litserve.server, "inference_worker", patched_inference_worker)

    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cpu", devices=1, timeout=5, max_batch_size=10)

    async with AsyncClient(server.app) as client:
        co1 = client.post("/predict", json={"input": 4.0})

    async with AsyncClient(server.app) as client:
        co2 = client.post("/predict", json={"input": 4.0})

    response1 = await co1
    response2 = await co2

    assert response1.json() == {"output": 9.0}
    assert response2.json() == {"output": 9.0}

    # TODO check that batch unbatch have been called
