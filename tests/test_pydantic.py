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
