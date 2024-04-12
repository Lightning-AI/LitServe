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
from litserve.server import LitServer
import pytest
from litserve.api import LitAPI
from fastapi import Request, Response
from fastapi.testclient import TestClient


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


@pytest.fixture()
def simple_litapi():
    return SimpleLitAPI()


@pytest.fixture()
def lit_server(simple_litapi):
    return LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)


@pytest.fixture()
def sync_testclient(lit_server):
    with TestClient(lit_server.app) as client:
        yield client
