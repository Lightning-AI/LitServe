# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from fastapi import Request, Response
from fastapi.testclient import TestClient

from pydantic import BaseModel

from litserve import LitAPI, LitServer


class PredictRequest(BaseModel):
    input: float

class PredictResponse(BaseModel):
    output: float

class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: PredictRequest) -> float:
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output: float) -> PredictResponse:
        return PredictResponse(output=output)


def test_pydantic():
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=5)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
