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
import subprocess
import time
from multiprocessing import Pipe, Manager
from asgi_lifespan import LifespanManager
from litserve import LitAPI
from fastapi import Request, Response

import torch
import torch.nn as nn
import os
from httpx import AsyncClient

from unittest.mock import patch, MagicMock

from litserve.connector import _Connector
from litserve.server import inference_worker, run_single_loop, run_streaming_loop
from litserve.server import LitServer

import pytest


def test_new_pipe(lit_server):
    pool_size = lit_server.max_pool_size
    for _ in range(pool_size):
        lit_server.new_pipe()

    assert len(lit_server.pipe_pool) == 0, "All available pipes from the pipe_pool were used up, which makes it empty"
    assert len(lit_server.new_pipe()) == 2, "lit_server.new_pipe() always must return a tuple of read and write pipes"


def test_dispose_pipe(lit_server):
    for i in range(lit_server.max_pool_size + 10):
        lit_server.dispose_pipe(*Pipe())
    assert len(lit_server.pipe_pool) == lit_server.max_pool_size, "pipe_pool size must be less than max_pool_size"


def test_index(sync_testclient):
    assert sync_testclient.get("/").text == "litserve running"


@patch("litserve.server.lifespan")
def test_device_identifiers(lifespan_mock, simple_litapi):
    server = LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)
    assert server.device_identifiers("cpu", 1) == ["cpu:1"]
    assert server.device_identifiers("cpu", [1, 2]) == ["cpu:1", "cpu:2"]

    server = LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)
    assert server.app.devices == ["cpu"]

    server = LitServer(simple_litapi, accelerator="cuda", devices=1, timeout=10)
    assert server.app.devices == [["cuda:0"]]

    server = LitServer(simple_litapi, accelerator="cuda", devices=[1, 2], timeout=10)
    # [["cuda:1"], ["cuda:2"]]
    assert server.app.devices[0][0] == "cuda:1"
    assert server.app.devices[1][0] == "cuda:2"


@patch("litserve.server.run_batched_loop")
@patch("litserve.server.run_single_loop")
def test_inference_worker(mock_single_loop, mock_batched_loop):
    inference_worker(*[MagicMock()] * 5, max_batch_size=2, batch_timeout=0, stream=False)
    mock_batched_loop.assert_called_once()

    inference_worker(*[MagicMock()] * 5, max_batch_size=1, batch_timeout=0, stream=False)
    mock_single_loop.assert_called_once()


@pytest.fixture()
def loop_args():
    from multiprocessing import Manager, Queue, Pipe

    requests_queue = Queue()
    request_buffer = Manager().dict()
    requests_queue.put(1)
    requests_queue.put(2)
    read, write = Pipe()
    request_buffer[1] = {"input": 4.0}, write
    request_buffer[2] = {"input": 5.0}, write

    lit_api_mock = MagicMock()
    lit_api_mock.decode_request = MagicMock(side_effect=lambda x: x["input"])
    return lit_api_mock, requests_queue, request_buffer


class FakePipe:
    def send(self, item):
        raise StopIteration("exit loop")


def test_single_loop(simple_litapi, loop_args):
    lit_api_mock, requests_queue, request_buffer = loop_args
    lit_api_mock.unbatch.side_effect = None
    request_buffer = Manager().dict()
    request_buffer[1] = {"input": 4.0}, FakePipe()
    request_buffer[2] = {"input": 5.0}, FakePipe()

    with pytest.raises(StopIteration, match="exit loop"):
        run_single_loop(lit_api_mock, requests_queue, request_buffer)


def test_run():
    process = subprocess.Popen(
        ["python", "tests/simple_server.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    time.sleep(5)
    assert os.path.exists("client.py"), f"Expected client file to be created at {os.getcwd()} after starting the server"
    output = subprocess.run("python client.py", shell=True, capture_output=True, text=True).stdout
    assert '{"output":16.0}' in output, "tests/simple_server.py didn't return expected output"
    os.remove("client.py")
    process.kill()


@pytest.mark.asyncio()
async def test_stream(simple_stream_api):
    server = LitServer(simple_stream_api, stream=True, timeout=10)
    expected_output1 = "prompt=Hello generated_output=LitServe is streaming output".lower().replace(" ", "")
    expected_output2 = "prompt=World generated_output=LitServe is streaming output".lower().replace(" ", "")

    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp1 = ac.post("/stream-predict", json={"prompt": "Hello"}, timeout=10)
        resp2 = ac.post("/stream-predict", json={"prompt": "World"}, timeout=10)
        resp1, resp2 = await asyncio.gather(resp1, resp2)
        assert resp1.status_code == 200, "Check if server is running and the request format is valid."
        assert resp1.text == expected_output1, "Server returns input prompt and generated output which didn't match."
        assert resp2.status_code == 200, "Check if server is running and the request format is valid."
        assert resp2.text == expected_output2, "Server returns input prompt and generated output which didn't match."


class FakeStreamPipe:
    def __init__(self, num_streamed_outputs):
        self.num_streamed_outputs = num_streamed_outputs
        self.count = 0

    def send(self, args):
        response, status = args
        if self.count >= self.num_streamed_outputs:
            raise StopIteration("exit loop")
        assert response == f"{self.count}", "This streaming loop generates number from 0 to 9 which is sent via Pipe"
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

    _, requests_queue, request_buffer = loop_args
    request_buffer = Manager().dict()
    request_buffer[1] = {"prompt": "Hello"}, FakeStreamPipe(num_streamed_outputs)

    with pytest.raises(StopIteration, match="exit loop"):
        run_streaming_loop(fake_stream_api, requests_queue, request_buffer)

    fake_stream_api.predict.assert_called_once_with("Hello")
    fake_stream_api.encode_response.assert_called_once()


def test_litapi_with_stream(simple_litapi):
    with pytest.raises(
        ValueError,
        match="""When `stream=True` both `lit_api.predict` and
             `lit_api.encode_response` must generate values using `yield.""",
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
