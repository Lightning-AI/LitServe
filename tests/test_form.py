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
from tests.conftest import wrap_litserve_start

from litserve import LitAPI, LitServer


class SimpleFileLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return len(request["input"].file.read().decode("utf-8"))

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_multipart_form_data(tmp_path):
    file_length = 1024 * 1024 * 100

    server = LitServer(
        SimpleFileLitAPI(), accelerator="cpu", devices=1, workers_per_device=1, max_payload_size=(file_length * 2)
    )

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        file_path = f"{tmp_path}/big_file.txt"
        with open(file_path, "wb") as f:
            f.write(bytearray([1] * file_length))
        with open(file_path, "rb") as f:
            file = {"input": f}
            response = client.post("/predict", files=file)
            assert response.json() == {"output": file_length**2}


def test_file_too_big(tmp_path):
    file_length = 1024 * 1024 * 100

    server = LitServer(
        SimpleFileLitAPI(), accelerator="cpu", devices=1, workers_per_device=1, max_payload_size=(file_length / 2)
    )

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        file_path = f"{tmp_path}/big_file.txt"
        with open(file_path, "wb") as f:
            f.write(bytearray([1] * file_length))
        with open(file_path, "rb") as f:
            file = {"input": f}
            response = client.post("/predict", files=file)
            assert response.status_code == 413

            # spoof content-length size
            response = client.post("/predict", files=file, headers={"content-length": "1024"})
            assert response.status_code == 413


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
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        file = {"input": "4.0"}
        response = client.post("/predict", data=file)
        assert response.json() == {"output": 16.0}
