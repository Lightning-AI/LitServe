from typing import Any, Union, List
from fastapi import Request, Response

from lit_server import LitAPI, LitServer

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


# vanilla API

class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request) -> Any:
        content = request["input"]
        return torch.tensor(content, device=self.device)[None, None]

    def predict(self, x: Any) -> Any:
        return self.model(x)

    def encode_response(self, output: Any) -> Response:
        return {"output": float(output)}


# with pydantic (advantage: FastAPI does schema validation)

from pydantic import BaseModel

class PredictRequest(BaseModel):
    input: float

class PredictResponse(BaseModel):
    output: float

class SimpleLitAPI2(LitAPI):
    def setup(self, device):
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: PredictRequest) -> float:
        content = request.input
        return torch.tensor(content, device=self.device)[None, None]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output: float) -> PredictResponse:
        return PredictResponse(output=float(output))


if __name__ == "__main__":
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=1, timeout=5)
    server.run(port=8000)
