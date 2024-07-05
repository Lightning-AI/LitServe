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
import inspect
import pickle
import re
from asgi_lifespan import LifespanManager
from litserve import LitAPI
from fastapi import Request, Response, HTTPException

import torch
import torch.nn as nn
from queue import Queue
from httpx import AsyncClient

from unittest.mock import patch, MagicMock
import pytest

from litserve.connector import _Connector
from litserve.server import (
    inference_worker,
    run_single_loop,
    run_streaming_loop,
    LitAPIStatus,
    run_batched_streaming_loop,
)
from litserve.server import LitServer
import litserve as ls
from fastapi.testclient import TestClient


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


@patch("litserve.server.run_batched_loop")
@patch("litserve.server.run_single_loop")
def test_inference_worker(mock_single_loop, mock_batched_loop):
    inference_worker(*[MagicMock()] * 6, max_batch_size=2, batch_timeout=0, stream=False)
    mock_batched_loop.assert_called_once()

    inference_worker(*[MagicMock()] * 6, max_batch_size=1, batch_timeout=0, stream=False)
    mock_single_loop.assert_called_once()


@pytest.fixture()
def loop_args():
    requests_queue = Queue()
    requests_queue.put(("uuid-123", 1))  # uid, x_enc
    requests_queue.put(("uuid-234", 2))

    lit_api_mock = MagicMock()
    lit_api_mock.decode_request = MagicMock(side_effect=lambda x: x["input"])
    return lit_api_mock, requests_queue


class FakeResponseQueue:
    def put(self, item):
        raise StopIteration("exit loop")


def test_single_loop(loop_args):
    lit_api_mock, requests_queue = loop_args
    lit_api_mock.unbatch.side_effect = None
    response_queue = FakeResponseQueue()

    with pytest.raises(StopIteration, match="exit loop"):
        run_single_loop(lit_api_mock, None, requests_queue, response_queue)


@pytest.mark.asyncio()
async def test_stream(simple_stream_api):
    server = LitServer(simple_stream_api, stream=True, timeout=10)
    expected_output1 = "prompt=Hello generated_output=LitServe is streaming output".lower().replace(" ", "")
    expected_output2 = "prompt=World generated_output=LitServe is streaming output".lower().replace(" ", "")

    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp1 = ac.post("/predict", json={"prompt": "Hello"}, timeout=10)
        resp2 = ac.post("/predict", json={"prompt": "World"}, timeout=10)
        resp1, resp2 = await asyncio.gather(resp1, resp2)
        assert resp1.status_code == 200, "Check if server is running and the request format is valid."
        assert resp1.text == expected_output1, "Server returns input prompt and generated output which didn't match."
        assert resp2.status_code == 200, "Check if server is running and the request format is valid."
        assert resp2.text == expected_output2, "Server returns input prompt and generated output which didn't match."


@pytest.mark.asyncio()
async def test_batched_stream_server(simple_batched_stream_api):
    server = LitServer(simple_batched_stream_api, stream=True, max_batch_size=4, batch_timeout=2, timeout=30)
    expected_output1 = "Hello LitServe is streaming output".lower().replace(" ", "")
    expected_output2 = "World LitServe is streaming output".lower().replace(" ", "")

    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp1 = ac.post("/predict", json={"prompt": "Hello"}, timeout=10)
        resp2 = ac.post("/predict", json={"prompt": "World"}, timeout=10)
        resp1, resp2 = await asyncio.gather(resp1, resp2)
        assert resp1.status_code == 200, "Check if server is running and the request format is valid."
        assert resp2.status_code == 200, "Check if server is running and the request format is valid."
        assert resp1.text == expected_output1, "Server returns input prompt and generated output which didn't match."
        assert resp2.text == expected_output2, "Server returns input prompt and generated output which didn't match."


class FakeStreamResponseQueue:
    def __init__(self, num_streamed_outputs):
        self.num_streamed_outputs = num_streamed_outputs
        self.count = 0

    def put(self, item):
        uid, args = item
        response, status = args
        if self.count >= self.num_streamed_outputs:
            raise StopIteration("exit loop")
        assert response == f"{self.count}", "This streaming loop generates number from 0 to 9 which is sent via Queue"
        self.count += 1


def test_streaming_loop(loop_args):
    num_streamed_outputs = 10

    def fake_predict(inputs: str):
        for i in range(num_streamed_outputs):
            yield {"output": f"{i}"}

    def fake_encode(output):
        assert inspect.isgenerator(output), "predict function must be a generator when `stream=True`"
        for out in output:
            yield out["output"]

    fake_stream_api = MagicMock()
    fake_stream_api.decode_request = MagicMock(side_effect=lambda x: x["prompt"])
    fake_stream_api.predict = MagicMock(side_effect=fake_predict)
    fake_stream_api.encode_response = MagicMock(side_effect=fake_encode)
    fake_stream_api.format_encoded_response = MagicMock(side_effect=lambda x: x)

    requests_queue = Queue()
    requests_queue.put(("UUID-1234", {"prompt": "Hello"}))
    response_queue = FakeStreamResponseQueue(num_streamed_outputs)

    with pytest.raises(StopIteration, match="exit loop"):
        run_streaming_loop(fake_stream_api, fake_stream_api, requests_queue, response_queue)

    fake_stream_api.predict.assert_called_once_with("Hello")
    fake_stream_api.encode_response.assert_called_once()


class FakeBatchStreamResponseQueue:
    def __init__(self, num_streamed_outputs):
        self.num_streamed_outputs = num_streamed_outputs
        self.count = 0

    def put(self, item):
        uid, args = item
        response, status = args
        if status == LitAPIStatus.FINISH_STREAMING:
            raise StopIteration("interrupt iteration")
        if status == LitAPIStatus.ERROR and b"interrupt iteration" in response:
            assert self.count // 2 == self.num_streamed_outputs, (
                f"Loop count must have incremented for " f"{self.num_streamed_outputs} times."
            )
            raise StopIteration("finish streaming")

        assert (
            response == f"{self.count // 2}"
        ), f"streaming loop generates number from 0 to 9 which is sent via Queue. {args}, count:{self.count}"
        self.count += 1


def test_batched_streaming_loop():
    num_streamed_outputs = 10

    def fake_predict(inputs: list):
        n = len(inputs)
        assert n == 2, "Two requests has been simulated to batched."
        for i in range(num_streamed_outputs):
            yield [{"output": f"{i}"}] * n

    def fake_encode(output_iter):
        assert inspect.isgenerator(output_iter), "predict function must be a generator when `stream=True`"
        for outputs in output_iter:
            yield [output["output"] for output in outputs]

    fake_stream_api = MagicMock()
    fake_stream_api.decode_request = MagicMock(side_effect=lambda x: x["prompt"])
    fake_stream_api.batch = MagicMock(side_effect=lambda inputs: inputs)
    fake_stream_api.predict = MagicMock(side_effect=fake_predict)
    fake_stream_api.encode_response = MagicMock(side_effect=fake_encode)
    fake_stream_api.unbatch = MagicMock(side_effect=lambda inputs: inputs)
    fake_stream_api.format_encoded_response = MagicMock(side_effect=lambda x: x)

    requests_queue = Queue()
    requests_queue.put(("UUID-001", {"prompt": "Hello"}))
    requests_queue.put(("UUID-002", {"prompt": "World"}))
    response_queue = FakeBatchStreamResponseQueue(num_streamed_outputs)

    with pytest.raises(StopIteration, match="finish streaming"):
        run_batched_streaming_loop(
            fake_stream_api, fake_stream_api, requests_queue, response_queue, max_batch_size=2, batch_timeout=2
        )
    fake_stream_api.predict.assert_called_once_with(["Hello", "World"])
    fake_stream_api.encode_response.assert_called_once()


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
    mock_uvicorn.run.assert_called()
    mock_uvicorn.reset_mock()
    server.run(port="8001")
    mock_uvicorn.run.assert_called()


class IndentityAPI(ls.examples.SimpleLitAPI):
    def predict(self, x, context):
        context["input"] = x
        return self.model(x)

    def encode_response(self, output, context):
        input = context["input"]
        return {"output": input}


class IndentityBatchedAPI(ls.examples.SimpleBatchedAPI):
    def predict(self, x_batch, context):
        for c, x in zip(context, x_batch):
            c["input"] = x
        return self.model(x_batch)

    def encode_response(self, output, context):
        input = context["input"]
        return {"output": input}


class IndentityBatchedStreamingAPI(ls.examples.SimpleBatchedAPI):
    def predict(self, x_batch, context):
        for c, x in zip(context, x_batch):
            c["input"] = x
        yield self.model(x_batch)

    def encode_response(self, output_stream, context):
        for _ in output_stream:
            yield [{"output": ctx["input"]} for ctx in context]


class PredictErrorAPI(ls.examples.SimpleLitAPI):
    def predict(self, x, y, context):
        context["input"] = x
        return self.model(x)

    def encode_response(self, output, context):
        input = context["input"]
        return {"output": input}


@pytest.mark.asyncio()
@patch("litserve.server.load_and_raise")
async def test_inject_context(mocked_load_and_raise):
    def dummy_load_and_raise(resp):
        raise pickle.loads(resp)

    mocked_load_and_raise.side_effect = dummy_load_and_raise

    # Test context injection with single loop
    api = IndentityAPI()
    server = LitServer(api)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    # Test context injection with batched loop
    server = LitServer(IndentityBatchedAPI(), max_batch_size=2, batch_timeout=0.01)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    # Test context injection with batched streaming loop
    server = LitServer(IndentityBatchedStreamingAPI(), max_batch_size=2, batch_timeout=0.01, stream=True)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)
    assert resp.json()["output"] == 5.0, "output from Identity server must be same as input"

    server = LitServer(PredictErrorAPI())
    with pytest.raises(TypeError, match=re.escape("predict() missing 1 required positional argument: 'y'")):
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            resp = await ac.post("/predict", json={"input": 5.0}, timeout=10)


def test_custom_api_path():
    with pytest.raises(ValueError, match="api_path must start with '/'. "):
        LitServer(ls.examples.SimpleLitAPI(), api_path="predict")

    server = LitServer(ls.examples.SimpleLitAPI(), api_path="/v1/custom_predict")
    url = server.api_path
    with TestClient(server.app) as client:
        response = client.post(url, json={"input": 4.0})
        assert response.status_code == 200, "Server response should be 200 (OK)"


class TestHTTPExceptionAPI(ls.examples.SimpleLitAPI):
    def decode_request(self, request):
        raise HTTPException(501, "decode request is bad")


def test_http_exception():
    server = LitServer(TestHTTPExceptionAPI())
    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 501, "Server raises 501 error"
        assert response.text == '{"detail":"decode request is bad"}', "decode request is bad"
