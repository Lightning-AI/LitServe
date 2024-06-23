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

from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.testclient import TestClient

from litserve import LitAPI, LitServer
import litserve.server


class SimpleAuthedLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if auth.scheme != "Bearer" or auth.credentials != "1234":
            raise HTTPException(status_code=401, detail="Bad token")


def test_authorized_custom():
    server = LitServer(SimpleAuthedLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with TestClient(server.app) as client:
        input = {"input": 4.0}
        response = client.post("/predict", headers={"Authorization": "Bearer 1234"}, json=input)
        assert response.status_code == 200


def test_not_authorized_custom():
    server = LitServer(SimpleAuthedLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with TestClient(server.app) as client:
        input = {"input": 4.0}
        response = client.post("/predict", headers={"Authorization": "Bearer wrong"}, json=input)
        assert response.status_code == 401


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_authorized_api_key():
    litserve.server.LIT_SERVER_API_KEY = "abcd"
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with TestClient(server.app) as client:
        input = {"input": 4.0}
        response = client.post("/predict", headers={"X-API-Key": "abcd"}, json=input)
        assert response.status_code == 200

    litserve.server.LIT_SERVER_API_KEY = None


def test_not_authorized_api_key():
    litserve.server.LIT_SERVER_API_KEY = "abcd"
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with TestClient(server.app) as client:
        input = {"input": 4.0}
        response = client.post("/predict", headers={"X-API-Key": "wrong"}, json=input)
        assert response.status_code == 401

    litserve.server.LIT_SERVER_API_KEY = None
