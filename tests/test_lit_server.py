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
import subprocess
from threading import Thread

import requests
import time
from multiprocessing import Pipe, Manager

import os

from unittest.mock import patch, MagicMock
from litserve.server import inference_worker, run_single_loop
from litserve.server import LitServer

import pytest


def test_new_pipe(lit_server):
    pool_size = lit_server.max_pool_size
    for _ in range(pool_size):
        lit_server.new_pipe()

    assert len(lit_server.pipe_pool) == 0
    assert len(lit_server.new_pipe()) == 2


def test_dispose_pipe(lit_server):
    for i in range(lit_server.max_pool_size + 10):
        lit_server.dispose_pipe(*Pipe())
    assert len(lit_server.pipe_pool) == lit_server.max_pool_size


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
    assert os.path.exists("client.py")
    output = subprocess.run("python client.py", shell=True, capture_output=True, text=True).stdout
    assert '{"output":16.0}' in output
    os.remove("client.py")
    process.kill()


def test_stream(simple_stream_api):
    server = LitServer(simple_stream_api, accelerator="cpu", stream=True)
    t = Thread(target=server.run, args=(), daemon=True)
    t.start()
    time.sleep(2)

    response = requests.post(
        "http://0.0.0.0:8000/stream-predict", json={"prompt": "Hello World"}, stream=True, timeout=5
    )

    output = "prompt=Hello World generated_output=LitServe is streaming output".split()
    assert response.status_code == 200
    for chunk, o in zip(response.iter_content(chunk_size=1024), output):
        assert chunk.decode("utf-8") == o.lower()
