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
from litserve.utils import wrap_litserve_start
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


class SimpleDelayedStreamAPI(SimpleStreamAPI):
    def encode_response(self, output: Generator) -> Generator:
        delay = 0.2
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


@pytest.fixture
def simple_litapi():
    return SimpleLitAPI()


@pytest.fixture
def simple_stream_api():
    return SimpleStreamAPI()


@pytest.fixture
def simple_batched_stream_api():
    return SimpleBatchedStreamAPI()


@pytest.fixture
def simple_delayed_stream_api():
    return SimpleDelayedStreamAPI()


@pytest.fixture
def lit_server(simple_litapi):
    server = LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)
    with wrap_litserve_start(server) as s:
        yield s


@pytest.fixture
def sync_testclient(lit_server):
    with TestClient(lit_server.app) as client:
        yield client


@pytest.fixture
def killall():
    def _run(process):
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        process.kill()

    return _run


@pytest.fixture
def openai_request_data():
    return {
        "model": "",
        "messages": [{"role": "string", "content": "string"}],
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "max_tokens": 0,
        "stop": "string",
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
    }


@pytest.fixture
def openai_response_data():
    return {
        "id": "chatcmpl-9dEtoQu4g45g3431SZ2s98S",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "10 + 6 is equal to 16.",
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1719139092,
        "model": "gpt-3.5-turbo-0125",
        "object": "chat.completion",
        "system_fingerprint": None,
        "usage": {"completion_tokens": 10, "prompt_tokens": 25, "total_tokens": 35},
    }


@pytest.fixture
def openai_request_data_with_image():
    return {
        "model": "lit",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                ],
            }
        ],
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "max_tokens": 0,
        "stop": "string",
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
    }


@pytest.fixture
def openai_request_data_with_tools():
    return {
        "model": "lit",
        "messages": [{"role": "user", "content": "What's the weather like in Boston today?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "max_tokens": 0,
        "stop": "string",
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
    }


@pytest.fixture
def openai_request_data_with_response_format():
    return {
        "model": "lit",
        "messages": [
            {
                "role": "system",
                "content": "Extract the event information.",
            },
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "calendar_event",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "date", "participants"],
                    "additionalProperties": "false",
                },
                "strict": "true",
            },
        },
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "max_tokens": 0,
        "stop": "string",
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
    }
