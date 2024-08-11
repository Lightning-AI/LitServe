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
from litserve.utils import wrap_litserve_start
from litserve import LitAPI, LitServer

# trivially compressible content
test_output = {"result": "0" * 100000}


class LargeOutputLitAPI(LitAPI):
    def setup(self, device):
        pass

    def decode_request(self, request: Request):
        pass

    def predict(self, x):
        pass

    def encode_response(self, output) -> Response:
        return test_output


def test_compression():
    server = LitServer(LargeOutputLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # compressed
        response = client.post("/predict", headers={"Accept-Encoding": "gzip"}, json={})
        assert response.status_code == 200
        assert response.headers["Content-Encoding"] == "gzip"
        content_length = int(response.headers["Content-Length"])
        assert 0 < content_length < 100000
        assert response.json() == test_output

        # uncompressed
        response = client.post("/predict", headers={"Accept-Encoding": ""}, json={})
        assert response.status_code == 200
        assert "Content-Encoding" not in response.headers
        content_length = int(response.headers["Content-Length"])
        assert content_length > 100000
        assert response.json() == test_output
