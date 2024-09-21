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
import pickle
import logging
import re
import sys

from asgi_lifespan import LifespanManager
from litserve import LitAPI
from fastapi import Request, Response, HTTPException
import torch
import torch.nn as nn
from httpx import AsyncClient
from litserve.utils import wrap_litserve_start
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from unittest.mock import patch, MagicMock
import pytest

from litserve.connector import _Connector

from litserve.server import LitServer
import litserve as ls
from fastapi.testclient import TestClient
from starlette.types import ASGIApp
from starlette.middleware.base import BaseHTTPMiddleware


def test_index(sync_testclient):
    assert sync_testclient.get("/").text == "litserve running"


@patch("litserve.server.LitServer.lifespan")
def test_device_identifiers(lifespan_mock, simple_litapi):
    server = LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)
    assert server.device_identifiers("cpu", 1) == ["cpu:1"]
    assert server.device_identifiers("cpu", [1, 2]) == ["cpu:1", "cpu:2"]

    server = LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)
    assert server.devices == ["cpu"]

    server = LitServer(simple_litapi, accelerator="cuda", devices=1, timeout=10)
    assert server.devices == [["cuda:0"]]

    server = LitServer(simple_litapi, accelerator="cuda", devices=[1, 2], timeout=10)
    # [["cuda:1"], ["cuda:2"]]
    assert server.devices[0][0] == "cuda:1"
    assert server.devices[1][0] == "cuda:2"


@pytest.mark.asyncio
async def test_stream(simple_stream_api):
    server = LitServer(simple_stream_api, stream=True, timeout=10)
    expected_output1 = "prompt=Hello generated_output=LitServe is streaming output".lower().replace(" ", "")
    expected_output2 = "prompt=World generated_output=LitServe is streaming output".lower().replace(" ", "")

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            resp1 = ac.post("/predict", json={"prompt": "Hello"}, timeout=10)
            resp2 = ac.post("/predict", json={"prompt": "World"}, timeout=10)
            resp1, resp2 = await asyncio.gather(resp1, resp2)
            assert resp1.status_code == 200, "Check if server is running and the request format is valid."
            assert (
                resp1.text == expected_output1
            ), "Server returns input prompt and generated output which didn't match."
            assert resp2.status_code == 200, "Check if server is running and the request format is valid."
            assert (
                resp2.text == expected_output2
            ), "Server returns input prompt and generated output which didn't match."


@pytest.mark.asyncio
async def test_stream_client_disconnection(simple_delayed_stream_api, caplog):
    server = LitServer(simple_delayed_stream_api, stream=True, timeout=10)

    with wrap_litserve_start(server) as server, caplog.at_level(logging.DEBUG):
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            task = asyncio.create_task(ac.post("/predict", json={"prompt": "Hey, How are you doing?"}, timeout=10))
            await asyncio.sleep(2)
            task.cancel()  # simulate client disconnection
            await asyncio.sleep(1)  # wait for the task to stop
            with pytest.raises(asyncio.CancelledError):
                await task
            assert "Streaming request cancelled for the uid=" in caplog.text
            # TODO: also check if the task actually stopped in the server

            caplog.clear()
            task = asyncio.create_task(ac.post("/predict", json={"prompt": "Hey, How are you doing?"}, timeout=10))
            await task
            assert "Streaming request cancelled for the uid=" not in caplog.text


@pytest.mark.asyncio
async def test_batched_stream_server(simple_batched_stream_api):
    server = LitServer(simple_batched_stream_api, stream=True, max_batch_size=4, batch_timeout=2, timeout=30)
    expected_output1 = "Hello LitServe is streaming output".lower().replace(" ", "")
    expected_output2 = "World LitServe is streaming output".lower().replace(" ", "")

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            resp1 = ac.post("/predict", json={"prompt": "Hello"}, timeout=10)
            resp2 = ac.post("/predict", json={"prompt": "World"}, timeout=10)
            resp1, resp2 = await asyncio.gather(resp1, resp2)
            assert resp1.status_code == 200, "Check if server is running and the request format is valid."
            assert resp2.status_code == 200, "Check if server is running and the request format is valid."
            assert (
                resp1.text == expected_output1
            ), "Server returns input prompt and generated output which didn't match."
            assert (
                resp2.text == expected_output2
            ), "Server returns input prompt and generated output which didn't match."


def test_litapi_with_stream(simple_litapi):
    with pytest.raises(
        ValueError,
        match="""When `stream=True` both `lit_api.predict` and
             `lit_api.encode_response` must generate values using `yield""",
    ):
        LitServer(simple_litapi, stream=True)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.fill_(2.0)
        self.linear.bias.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request):
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def predict(self, x):
        return self.model(x[None, :])

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


@pytest.mark.parametrize(
    ("input_accelerator", "expected_accelerator"),
    [
        ("cpu", "cpu"),
        pytest.param(
            "cuda",
            "cuda",
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            None, "cuda", marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU")
        ),
        pytest.param(
            "auto",
            "cuda",
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            "auto",
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
        pytest.param(
            None,
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
    ],
)
def test_auto_accelerator(input_accelerator, expected_accelerator):
    server = LitServer(SimpleLitAPI(), devices=1, timeout=10, accelerator=input_accelerator)
    assert server._connector.accelerator == expected_accelerator


def test_mocked_accelerator():
    # 1. cuda available
    with patch("litserve.connector.check_cuda_with_nvidia_smi", return_value=True):
        connector = _Connector(accelerator="auto")
        assert connector.accelerator == "cuda"

    # 2. mps available
    with patch("litserve.connector._Connector._choose_gpu_accelerator_backend", return_value="mps"):
        server = LitServer(SimpleLitAPI(), devices=1, timeout=10, accelerator="auto")
        assert server._connector.accelerator == "mps"


@patch("litserve.server.uvicorn")
def test_server_run(mock_uvicorn):
    server = LitServer(SimpleLitAPI())
    with pytest.raises(ValueError, match="port must be a value from 1024 to 65535 but got"):
        server.run(port="invalid port")

    with pytest.raises(ValueError, match="port must be a value from 1024 to 65535 but got"):
        server.run(port=65536)

    server.run(port=8000)
    mock_uvicorn.Config.assert_called()
    mock_uvicorn.reset_mock()
    server.run(port="8001")
    mock_uvicorn.Config.assert_called()


@pytest.mark.skipif(sys.platform == "win32", reason="Test is only for Unix")
@patch("litserve.server.uvicorn")
def test_start_server(mock_uvicon):
    server = LitServer(ls.test_examples.TestAPI(), spec=ls.OpenAISpec())
    sockets = MagicMock()
    server._start_server(8000, 1, "info", sockets, "process")
    mock_uvicon.Server.assert_called()
    assert server.lit_spec.response_queue_id is not None, "response_queue_id must be generated"


@pytest.mark.skipif(sys.platform == "win32", reason="Test is only for Unix")
@patch("litserve.server.uvicorn")
def test_server_run_with_api_server_worker_type(mock_uvicorn):
    api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(api, devices=1)
    with pytest.raises(ValueError, match=r"Must be 'process' or 'thread'"):
        server.run(api_server_worker_type="invalid")

    with pytest.raises(ValueError, match=r"must be greater than 0"):
        server.run(num_api_servers=0)

    server.launch_inference_worker = MagicMock(return_value=[MagicMock(), [MagicMock()]])
    server._start_server = MagicMock()

    # Running the method to test
    server.run(api_server_worker_type=None)
    server.launch_inference_worker.assert_called_with(1)
    actual = server._start_server.call_args
    assert actual[0][4] == "process", "Server should run in process mode"

    server.run(api_server_worker_type="thread")
    server.launch_inference_worker.assert_called_with(1)
    actual = server._start_server.call_args
    assert actual[0][4] == "thread", "Server should run in thread mode"

    server.run(api_server_worker_type="process")
    server.launch_inference_worker.assert_called_with(1)
    actual = server._start_server.call_args
    assert actual[0][4] == "process", "Server should run in process mode"

    server.run(api_server_worker_type="process", num_api_servers=10)
    server.launch_inference_worker.assert_called_with(10)


@pytest.mark.skipif(sys.platform != "win32", reason="Test is only for Windows")
@patch("litserve.server.uvicorn")
def test_server_run_windows(mock_uvicorn):
    api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(api)
    server.launch_inference_worker = MagicMock(return_value=[MagicMock(), [MagicMock()]])
    server._start_server = MagicMock()

    # Running the method to test
    server.run(api_server_worker_type=None)
    actual = server._start_server.call_args
    assert actual[0][4] == "thread", "Windows only supports thread mode"


def test_server_terminate():
    server = LitServer(SimpleLitAPI())
    mock_manager = MagicMock()

    with patch("litserve.server.LitServer._start_server", side_effect=Exception("mocked error")) as mock_start, patch(
        "litserve.server.LitServer.launch_inference_worker", return_value=[mock_manager, [MagicMock()]]
    ) as mock_launch:
        with pytest.raises(Exception, match="mocked error"):
            server.run(port=8001)

        mock_launch.assert_called()
        mock_start.assert_called()
        mock_manager.shutdown.assert_called()


class IdentityAPI(ls.test_examples.SimpleLitAPI):
    def predict(self, x, context):
        context["input"] = x
        return self.model(x)

    def encode_response(self, output, context):
        input = context["input"]
        return {"output": input}


class IdentityBatchedAPI(ls.test_examples.SimpleBatchedAPI):
    def predict(self, x_batch, context):
        for c, x in zip(context, x_batch):
            c["input"] = x
        return self.model(x_batch)

    def encode_response(self, output, context):
        input = context["input"]
        return {"output": input}


class IdentityBatchedStreamingAPI(ls.test_examples.SimpleBatchedAPI):
    def predict(self, x_batch, context):
        for c, x in zip(context, x_batch):
            c["input"] = x
        yield self.model(x_batch)

    def encode_response(self, output_stream, context):
        for _ in output_stream:
            yield [{"output": ctx["input"]} for ctx in context]


class PredictErrorAPI(ls.test_examples.SimpleLitAPI):
    def predict(self, x, y, context):
        context["input"] = x
        return self.model(x)

    def encode_response(self, output, context):
        input = context["input"]
        return {"output": input}


@pytest.mark.asyncio
@patch("litserve.server.load_and_raise")
async def test_inject_context(mocked_load_and_raise):
    def dummy_load_and_raise(resp):
        raise pickle.loads(resp)

    mocked_load_and_raise.side_effect = dummy_load_and_raise

    # Test context injection with single loop
    api = IdentityAPI()
    server = LitServer(api)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    # Test context injection with batched loop
    server = LitServer(IdentityBatchedAPI(), max_batch_size=2, batch_timeout=0.01)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    # Test context injection with batched streaming loop
    server = LitServer(IdentityBatchedStreamingAPI(), max_batch_size=2, batch_timeout=0.01, stream=True)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    server = LitServer(PredictErrorAPI())
    with wrap_litserve_start(server) as server, pytest.raises(
        TypeError, match=re.escape("predict() missing 1 required positional argument: 'y'")
    ), TestClient(server.app) as client:
        client.post("/predict", json={"input": 5.0}, timeout=10)


def test_custom_api_path():
    with pytest.raises(ValueError, match="api_path must start with '/'. "):
        LitServer(ls.test_examples.SimpleLitAPI(), api_path="predict")

    server = LitServer(ls.test_examples.SimpleLitAPI(), api_path="/v1/custom_predict")
    url = server.api_path
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post(url, json={"input": 4.0})
        assert response.status_code == 200, "Server response should be 200 (OK)"


class TestHTTPExceptionAPI(ls.test_examples.SimpleLitAPI):
    def decode_request(self, request):
        raise HTTPException(501, "decode request is bad")


def test_http_exception():
    server = LitServer(TestHTTPExceptionAPI())
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 501, "Server raises 501 error"
        assert response.text == '{"detail":"decode request is bad"}', "decode request is bad"


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, length: int) -> None:
        self.app = app
        self.length = length
        super().__init__(app)

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Request-Id"] = "0" * self.length
        return response


def test_custom_middleware():
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=[(RequestIdMiddleware, {"length": 5})])
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 200, f"Expected response to be 200 but got {response.status_code}"
        assert response.json() == {"output": 16.0}, "server didn't return expected output"
        assert response.headers["X-Request-Id"] == "00000"


def test_starlette_middlewares():
    middlewares = [
        (
            TrustedHostMiddleware,
            {
                "allowed_hosts": ["localhost", "127.0.0.1"],
            },
        ),
        HTTPSRedirectMiddleware,
    ]
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=middlewares)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0}, headers={"Host": "localhost"})
        assert response.status_code == 200, f"Expected response to be 200 but got {response.status_code}"
        assert response.json() == {"output": 16.0}, "server didn't return expected output"

        response = client.post("/predict", json={"input": 4.0}, headers={"Host": "not-trusted-host"})
        assert response.status_code == 400, f"Expected response to be 400 but got {response.status_code}"


def test_middlewares_inputs():
    server = ls.LitServer(SimpleLitAPI(), middlewares=[])
    assert len(server.middlewares) == 1, "Default middleware should be present"

    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=[], max_payload_size=1000)
    assert len(server.middlewares) == 2, "Default middleware should be present"

    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=None)
    assert len(server.middlewares) == 1, "Default middleware should be present"

    with pytest.raises(ValueError, match="middlewares must be a list of tuples"):
        ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=(RequestIdMiddleware, {"length": 5}))
