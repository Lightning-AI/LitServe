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
from fastapi import Request, Response
from fastapi.testclient import TestClient

from litserve import LitAPI, LitServer

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

    def predict(self, x):
        return self.model(x[None, :])

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


def test_torch():
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=10)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 9.0}


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="requires CUDA to be available")
def test_torch_gpu():
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=1, timeout=10)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 9.0}
