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


class SimpleFileLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return float(request["input"].file.read().decode("utf-8"))

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_multipart_form_data():
    server = LitServer(SimpleFileLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with TestClient(server.app) as client:
        file = {"input": "4.0"}
        response = client.post("/predict", files=file)
        assert response.json() == {"output": 16.0}


class SimpleFormLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return float(request["input"])

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_urlencoded_form_data():
    server = LitServer(SimpleFormLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with TestClient(server.app) as client:
        file = {"input": "4.0"}
        response = client.post("/predict", data=file)
        assert response.json() == {"output": 16.0}
