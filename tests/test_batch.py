# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from fastapi import Request, Response
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor

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

    def predict(self, x):
        x = torch.stack(x)
        assert x.shape == (2, 1), f"{x.shape}"
        output = self.model(x)
        output = list(output)
        return output

    def encode_response(self, output) -> Response:
        assert len(output) == 1
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
        output = self.model(x)
        return output

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


def test_batched():
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cpu", devices=1, timeout=2, max_batch_size=10, batch_timeout=1)

    with ThreadPoolExecutor(2) as executor, TestClient(server.app) as client:
        response1 = executor.submit(client.post, "/predict", json={"input": 4.0})
        response2 = executor.submit(client.post, "/predict", json={"input": 5.0})

    assert response1.result().json() == {"output": 9.0}
    assert response2.result().json() == {"output": 11.0}

    # TODO check that batch unbatch have been called


def test_unbatched():
    api = SimpleLitAPI2()
    server = LitServer(api, accelerator="cpu", devices=1, timeout=2, max_batch_size=1, batch_timeout=0)

    with ThreadPoolExecutor(2) as executor, TestClient(server.app) as client:
        response1 = executor.submit(client.post, "/predict", json={"input": 4.0})
        response2 = executor.submit(client.post, "/predict", json={"input": 5.0})

    assert response1.result().json() == {"output": 9.0}
    assert response2.result().json() == {"output": 11.0}
