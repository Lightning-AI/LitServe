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
import sys
from time import sleep
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from asgi_lifespan import LifespanManager
from fastapi import HTTPException, Request, Response
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve import LitAPI
from litserve.connector import _Connector
from litserve.server import LitServer, multi_server_lifespan, run_all
from litserve.test_examples.openai_spec_example import TestAPI
from litserve.utils import wrap_litserve_start


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


@pytest.mark.parametrize("devices", ["cpu", ["cpu", "cuda:0"], ["cuda:a", "cuda:1"]])
def test_device_identifiers_error(simple_litapi, devices):
    with pytest.raises(
        ValueError, match="devices must be an integer or a list of integers when using 'cuda' or 'mps', instead got .*"
    ):
        LitServer(simple_litapi, accelerator="cuda", devices=devices, timeout=10)


@pytest.mark.parametrize("use_zmq", [True, False])
@pytest.mark.asyncio
async def test_stream(simple_stream_api, use_zmq):
    server = LitServer(simple_stream_api, stream=True, timeout=10, use_zmq=use_zmq)
    expected_output1 = "prompt=Hello generated_output=LitServe is streaming output".lower().replace(" ", "")
    expected_output2 = "prompt=World generated_output=LitServe is streaming output".lower().replace(" ", "")

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            # TODO: remove this sleep when we have a better way to check if the server is ready
            # TODO: main process can only consume when response_queue_to_buffer is ready
            await asyncio.sleep(4)
            resp1 = ac.post("/predict", json={"prompt": "Hello"}, timeout=10)
            resp2 = ac.post("/predict", json={"prompt": "World"}, timeout=10)
            resp1, resp2 = await asyncio.gather(resp1, resp2)
            assert resp1.status_code == 200, "Check if server is running and the request format is valid."
            assert resp1.text == expected_output1, (
                "Server returns input prompt and generated output which didn't match."
            )
            assert resp2.status_code == 200, "Check if server is running and the request format is valid."
            assert resp2.text == expected_output2, (
                "Server returns input prompt and generated output which didn't match."
            )


@pytest.mark.parametrize("use_zmq", [True, False])
@pytest.mark.asyncio
async def test_batched_stream_server(simple_batched_stream_api, use_zmq):
    server = LitServer(
        simple_batched_stream_api, stream=True, max_batch_size=4, batch_timeout=2, timeout=30, use_zmq=use_zmq
    )
    expected_output1 = "Hello LitServe is streaming output".lower().replace(" ", "")
    expected_output2 = "World LitServe is streaming output".lower().replace(" ", "")

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp1 = ac.post("/predict", json={"prompt": "Hello"}, timeout=10)
            resp2 = ac.post("/predict", json={"prompt": "World"}, timeout=10)
            resp1, resp2 = await asyncio.gather(resp1, resp2)
            assert resp1.status_code == 200, "Check if server is running and the request format is valid."
            assert resp2.status_code == 200, "Check if server is running and the request format is valid."
            assert resp1.text == expected_output1, (
                "Server returns input prompt and generated output which didn't match."
            )
            assert resp2.text == expected_output2, (
                "Server returns input prompt and generated output which didn't match."
            )


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
    server.verify_worker_status = MagicMock()
    with pytest.raises(ValueError, match="port must be a value from 1024 to 65535 but got"):
        server.run(port="invalid port")

    with pytest.raises(ValueError, match="port must be a value from 1024 to 65535 but got"):
        server.run(port=65536)

    with pytest.raises(ValueError, match="host must be '0.0.0.0', '127.0.0.1', or '::' but got"):
        server.run(host="127.0.0.2")

    server.run(port=8000)
    mock_uvicorn.Config.assert_called()
    mock_uvicorn.reset_mock()
    server.run(port="8001")
    mock_uvicorn.Config.assert_called()
    server.run(host="::", port="8000")
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
    server.verify_worker_status = MagicMock()
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
    server.verify_worker_status = MagicMock()
    server.launch_inference_worker = MagicMock(return_value=[MagicMock(), [MagicMock()]])
    server._start_server = MagicMock()

    # Running the method to test
    server.run(api_server_worker_type=None)
    actual = server._start_server.call_args
    assert actual[0][4] == "thread", "Windows only supports thread mode"


def test_server_terminate():
    server = LitServer(SimpleLitAPI())
    server.verify_worker_status = MagicMock()
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
async def test_inject_context():
    # Test context injection with single loop
    api = IdentityAPI()
    server = LitServer(api)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    # Test context injection with batched loop
    server = LitServer(IdentityBatchedAPI(), max_batch_size=2, batch_timeout=0.01)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    # Test context injection with batched streaming loop
    server = LitServer(IdentityBatchedStreamingAPI(), max_batch_size=2, batch_timeout=0.01, stream=True)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    server = LitServer(PredictErrorAPI())
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        resp = client.post("/predict", json={"input": 5.0}, timeout=10)
        assert resp.status_code == 500, "predict() missed 1 required positional argument: 'y'"


def test_custom_api_path():
    with pytest.raises(ValueError, match="api_path must start with '/'. "):
        LitServer(ls.test_examples.SimpleLitAPI(), api_path="predict")

    server = LitServer(ls.test_examples.SimpleLitAPI(), api_path="/v1/custom_predict")
    url = server.api_path
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post(url, json={"input": 4.0})
        assert response.status_code == 200, "Server response should be 200 (OK)"


def test_custom_healthcheck_path():
    with pytest.raises(ValueError, match="healthcheck_path must start with '/'. "):
        LitServer(ls.test_examples.SimpleLitAPI(), healthcheck_path="customhealth")

    server = LitServer(ls.test_examples.SimpleLitAPI(), healthcheck_path="/v1/custom_health")
    url = server.healthcheck_path
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Sleep a bit to ensure the server is ready
        sleep(3)
        response = client.get(url)

    assert response.status_code == 200, "Server response should be 200 (OK)"


def test_custom_info_path():
    with pytest.raises(ValueError, match="info_path must start with '/'. "):
        LitServer(ls.test_examples.SimpleLitAPI(), info_path="custominfo")

    server = LitServer(ls.test_examples.SimpleLitAPI(), info_path="/v1/custom_info", accelerator="cpu")
    url = server.info_path
    expected_response = {
        "model": None,
        "server": {
            "devices": ["cpu"],
            "workers_per_device": 1,
            "timeout": 30,
            "max_batch_size": 1,
            "batch_timeout": 0.0,
            "stream": False,
            "max_payload_size": None,
            "track_requests": False,
        },
    }

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Sleep a bit to ensure the server is ready
        sleep(3)
        response = client.get(url)

    assert response.status_code == 200, "Server response should be 200 (OK)"
    assert response.json() == expected_response, "server didn't return expected output"


def test_info_route():
    model_metadata = {"name": "my-awesome-model", "version": "v1.1.0"}
    expected_response = {
        "model": {
            "name": "my-awesome-model",
            "version": "v1.1.0",
        },
        "server": {
            "devices": ["cpu"],
            "workers_per_device": 1,
            "timeout": 30,
            "max_batch_size": 1,
            "batch_timeout": 0.0,
            "stream": False,
            "max_payload_size": None,
            "track_requests": False,
        },
    }

    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), accelerator="cpu", model_metadata=model_metadata)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.get("/info", headers={"Host": "localhost"})
        assert response.status_code == 200, f"Expected response to be 200 but got {response.status_code}"
        assert response.json() == expected_response, "server didn't return expected output"


def test_model_metadata_json_error():
    with pytest.raises(ValueError, match="model_metadata must be JSON serializable"):
        ls.LitServer(ls.test_examples.SimpleLitAPI(), model_metadata=int)


class TestHTTPExceptionAPI(ls.test_examples.SimpleLitAPI):
    def decode_request(self, request):
        raise HTTPException(501, "decode request is bad")


class TestHTTPExceptionAPI2(ls.test_examples.SimpleLitAPI):
    def decode_request(self, request):
        raise HTTPException(status_code=400, detail="decode request is bad")


def test_http_exception():
    server = LitServer(TestHTTPExceptionAPI())
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 501, "Server raises 501 error"
        assert response.text == '{"detail":"decode request is bad"}', "decode request is bad"

    server = LitServer(TestHTTPExceptionAPI2())
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 400, "Server raises 400 error"
        assert response.text == '{"detail":"decode request is bad"}', "decode request is bad"


def test_generate_client_file(tmp_path, monkeypatch):
    expected = """import requests

response = requests.post("http://127.0.0.1:8123/predict", json={"input": 4.0})
print(f"Status: {response.status_code}\\nResponse:\\n {response.text}")"""
    monkeypatch.chdir(tmp_path)
    LitServer.generate_client_file(8123)
    with open(tmp_path / "client.py") as fr:
        assert expected in fr.read(), f"Expected {expected} in client.py"

    LitServer.generate_client_file(8000)
    with open(tmp_path / "client.py") as fr:
        assert expected in fr.read(), "Shouldn't replace existing client.py"


class FailFastAPI(ls.test_examples.SimpleLitAPI):
    def setup(self, device):
        raise ValueError("setup failed")


@pytest.mark.asyncio
@patch("litserve.server.LitServer")
async def test_multi_server_lifespan(mock_litserver):
    # List of servers
    servers = [mock_litserver, mock_litserver]
    # Use the async context manager
    async with multi_server_lifespan(MagicMock(), servers):
        # Check if the lifespan method was called for each server
        assert mock_litserver.lifespan.call_count == 2
    assert mock_litserver.lifespan.return_value.__aexit__.call_count == 2


@patch("litserve.server.uvicorn")
def test_run_all_litservers(mock_uvicorn):
    server1 = LitServer(SimpleLitAPI(), api_path="/predict-1")
    server2 = LitServer(SimpleLitAPI(), api_path="/predict-2")
    server3 = LitServer(TestAPI(), spec=ls.OpenAISpec())

    with pytest.raises(ValueError, match="All elements in the servers list must be instances of LitServer"):
        run_all([server1, "server2"])

    with pytest.raises(ValueError, match="port must be a value from 1024 to 65535 but got"):
        run_all([server1, server2], port="invalid port")

    with pytest.raises(ValueError, match="port must be a value from 1024 to 65535 but got"):
        run_all([server1, server2], port=65536)

    with pytest.raises(ValueError, match="num_api_servers must be greater than 0"):
        run_all([server1, server2], num_api_servers=0)

    run_all([server1, server2, server3], port=8000)
    mock_uvicorn.Config.assert_called()
    mock_uvicorn.reset_mock()
    run_all([server1, server2, server3], port="8001")
    mock_uvicorn.Config.assert_called()


def test_workers_setup_status():
    api = FailFastAPI()
    server = LitServer(api, devices=1)
    with pytest.raises(RuntimeError, match="One or more workers failed to start. Shutting down LitServe"):
        server.run()
