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
import time
import psutil
from typing import Generator

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


class SimpleStreamAPI(LitAPI):
    def setup(self, device) -> None:
        self.sentence = "LitServe is streaming output"

    def decode_request(self, request: Request) -> str:
        return request["prompt"]

    def predict(self, x) -> Generator:
        output = f"prompt={x} generated_output={self.sentence}".split()
        yield from output

    def encode_response(self, output: Generator) -> Generator:
        delay = 0.01  # delay for testing timeouts
        for out in output:
            time.sleep(delay)
            yield out.lower()


class SimpleBatchedStreamAPI(LitAPI):
    def setup(self, device) -> None:
        self.sentence = "LitServe is streaming output"

    def decode_request(self, request: Request) -> str:
        return request["prompt"]

    def batch(self, inputs):
        return inputs

    def predict(self, x) -> Generator:
        n = len(x)
        output = self.sentence.split()
        responses = [x]
        for out in output:
            responses.append([out] * n)
        yield from responses

    def encode_response(self, output: Generator) -> Generator:
        delay = 0.01  # delay for testing timeouts
        for out in output:
            time.sleep(delay)
            yield [e.lower() for e in out]

    def unbatch(self, output):
        yield from output


@pytest.fixture()
def simple_litapi():
    return SimpleLitAPI()


@pytest.fixture()
def simple_stream_api():
    return SimpleStreamAPI()


@pytest.fixture()
def simple_batched_stream_api():
    return SimpleBatchedStreamAPI()


@pytest.fixture()
def lit_server(simple_litapi):
    return LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)


@pytest.fixture()
def sync_testclient(lit_server):
    with TestClient(lit_server.app) as client:
        yield client


@pytest.fixture()
def killall():
    def _run(process):
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        process.kill()

    return _run


@pytest.fixture()
def openai_request_data():
    return {
        "model": "",
        "messages": "string",
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
    }
