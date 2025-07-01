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
import json
from queue import Queue
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from litserve.server import BaseRequestHandler, RegularRequestHandler
from litserve.test_examples import SimpleLitAPI
from litserve.utils import LitAPIStatus


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
    mock_server.response_buffer.pop.return_value = ("test-response", LitAPIStatus.OK)
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
