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
import asyncio
import json
import pickle
from queue import Queue
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from litserve import LitAPI, LitServer
from litserve.loops.base import _async_inject_context, _inject_context
from litserve.loops.continuous_batching_loop import DefaultContinuousBatchingLoop
from litserve.server import BaseRequestHandler, RegularRequestHandler
from litserve.test_examples import SimpleLitAPI
from litserve.utils import LitAPIStatus, ResponseBufferItem


@pytest.fixture
def mock_lit_api():
    return SimpleLitAPI()


class MockServer:
    def __init__(self, lit_api):
        self.lit_api = lit_api
        self.response_buffer = {}
        self.request_queue = Queue()
        self._callback_runner = mock.MagicMock()
        self.app = mock.MagicMock()
        self.app.response_queue_id = 0
        self.active_requests = 0

    def _get_request_queue(self, api_path):
        return self.request_queue


class MockRequest:
    """Mock FastAPI Request object for testing."""

    def __init__(self, json_data=None, form_data=None, content_type="application/json"):
        self._json_data = json_data or {}
        self._form_data = form_data or {}
        self.headers = {"Content-Type": content_type}

    async def json(self):
        if self._json_data is None:
            raise json.JSONDecodeError("Invalid JSON", "", 0)
        return self._json_data

    async def form(self):
        return self._form_data


class TestRequestHandler(BaseRequestHandler):
    def __init__(self, lit_api, server):
        super().__init__(lit_api, server)
        self.litapi_request_queues = {"/predict": Queue()}

    async def handle_request(self, request, request_type):
        payload = await self._prepare_request(request, request_type)
        uid, response_queue_id = await self._submit_request(payload)
        return response_queue_id


@pytest.mark.asyncio
async def test_request_handler(mock_lit_api):
    mock_server = MockServer(mock_lit_api)
    handler = TestRequestHandler(mock_lit_api, mock_server)
    mock_request = MockRequest()
    response_queue_id = await handler.handle_request(mock_request, Request)
    assert response_queue_id == 0


@pytest.mark.asyncio
@patch("litserve.server.asyncio.Event")
async def test_request_handler_streaming(mock_event, mock_lit_api):
    mock_event.return_value = AsyncMock()
    mock_server = MockServer(mock_lit_api)
    mock_request = MockRequest()
    mock_server.response_buffer = MagicMock()
    mock_server.response_buffer.pop.return_value = ResponseBufferItem(
        asyncio.Event(), response=("test-response", LitAPIStatus.OK)
    )
    handler = RegularRequestHandler(mock_lit_api, mock_server)
    response = await handler.handle_request(mock_request, Request)
    assert mock_server.request_queue.qsize() == 1
    assert response == "test-response"


def test_regular_handler_error_response():
    with pytest.raises(HTTPException) as e:
        RegularRequestHandler._handle_error_response(HTTPException(status_code=500, detail="test error response"))
    assert e.value.status_code == 500
    assert e.value.detail == "test error response"

    with pytest.raises(HTTPException) as e:
        RegularRequestHandler._handle_error_response(Exception("test exception"))
    assert e.value.status_code == 500
    assert e.value.detail == "Internal server error"


def make_request(body, content_type):
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", content_type.encode())] if content_type else []
    return Request({"type": "http", "headers": headers}, receive)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"\x00\xff\x80", b"", b'{"input": 4}'])
@pytest.mark.parametrize("content_type", ["application/x-recordio-protobuf", "application/json; charset=utf-8", ""])
async def test_raw_request_transport(mock_lit_api, body, content_type):
    server = MockServer(mock_lit_api)
    handler = TestRequestHandler(mock_lit_api, server)
    await handler.handle_request(make_request(body, content_type), bytes)
    request_data = pickle.loads(pickle.dumps(server.request_queue.get_nowait()))
    assert len(request_data) == 4
    payload = request_data[3]

    def decode(request, context):
        assert type(request) is bytes
        assert request == body
        assert context == {"content_type": content_type}
        return request

    assert _inject_context({}, decode, payload) == body
    assert await _async_inject_context({}, decode, payload) == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    ["application/json", "application/json; charset=utf-8", "application/vnd.api+json", "text/plain", ""],
)
async def test_default_request_still_parses_json(mock_lit_api, content_type):
    handler = TestRequestHandler(mock_lit_api, MockServer(mock_lit_api))
    request = make_request(b'{"input": 4}', content_type)
    assert await handler._prepare_request(request, Request) == {"input": 4}


@pytest.mark.asyncio
async def test_default_request_does_not_fall_back_to_bytes(mock_lit_api):
    handler = TestRequestHandler(mock_lit_api, MockServer(mock_lit_api))
    request = make_request(b"\xff\x00", "application/x-recordio-protobuf")
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        await handler._prepare_request(request, Request)


class BinaryRequestAPI(LitAPI):
    def decode_request(self, request: bytes, context):
        assert type(request) is bytes
        return request, context["content_type"]


class ForwardAnnotatedBinaryRequestAPI(BinaryRequestAPI):
    def decode_request(self, request: "bytes", context) -> "ModelInput":  # noqa: F821
        return super().decode_request(request, context)


@pytest.mark.parametrize("api_cls", [BinaryRequestAPI, ForwardAnnotatedBinaryRequestAPI])
def test_binary_request_openapi(api_cls):
    server = LitServer(api_cls(), accelerator="cpu", devices=1)
    operation = server.app.openapi()["paths"]["/predict"]["post"]
    assert not any(parameter["name"] == "request" for parameter in operation.get("parameters", []))
    assert operation["requestBody"]["content"]["application/octet-stream"]["schema"] == {
        "type": "string",
        "format": "binary",
    }


@pytest.mark.asyncio
async def test_continuous_batching_decodes_raw_request():
    api = BinaryRequestAPI()
    handler = TestRequestHandler(api, MockServer(api))
    payload = await handler._prepare_request(make_request(b"\xff\x00", "application/x-recordio-protobuf"), bytes)
    loop = DefaultContinuousBatchingLoop()
    loop.add_request("request-1", payload, api, None)
    assert loop.active_sequences["request-1"]["input"] == (b"\xff\x00", "application/x-recordio-protobuf")
