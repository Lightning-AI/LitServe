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
import inspect

import time
from queue import Queue

from unittest.mock import MagicMock, patch
import pytest

from litserve.loops import (
    run_single_loop,
    run_streaming_loop,
    run_batched_streaming_loop,
    inference_worker,
)
from litserve.utils import LitAPIStatus


@pytest.fixture()
def loop_args():
    requests_queue = Queue()
    requests_queue.put((0, "uuid-123", time.monotonic(), 1))  # response_queue_id, uid, timestamp, x_enc
    requests_queue.put((1, "uuid-234", time.monotonic(), 2))

    lit_api_mock = MagicMock()
    lit_api_mock.request_timeout = 1
    lit_api_mock.decode_request = MagicMock(side_effect=lambda x: x["input"])
    return lit_api_mock, requests_queue


class FakeResponseQueue:
    def put(self, item):
        raise StopIteration("exit loop")


def test_single_loop(loop_args):
    lit_api_mock, requests_queue = loop_args
    lit_api_mock.unbatch.side_effect = None
    response_queues = [FakeResponseQueue()]

    with pytest.raises(StopIteration, match="exit loop"):
        run_single_loop(lit_api_mock, None, requests_queue, response_queues)


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


def test_streaming_loop():
    num_streamed_outputs = 10

    def fake_predict(inputs: str):
        for i in range(num_streamed_outputs):
            yield {"output": f"{i}"}

    def fake_encode(output):
        assert inspect.isgenerator(output), "predict function must be a generator when `stream=True`"
        for out in output:
            yield out["output"]

    fake_stream_api = MagicMock()
    fake_stream_api.request_timeout = 1
    fake_stream_api.decode_request = MagicMock(side_effect=lambda x: x["prompt"])
    fake_stream_api.predict = MagicMock(side_effect=fake_predict)
    fake_stream_api.encode_response = MagicMock(side_effect=fake_encode)
    fake_stream_api.format_encoded_response = MagicMock(side_effect=lambda x: x)

    requests_queue = Queue()
    requests_queue.put((0, "UUID-1234", time.monotonic(), {"prompt": "Hello"}))
    response_queues = [FakeStreamResponseQueue(num_streamed_outputs)]

    with pytest.raises(StopIteration, match="exit loop"):
        run_streaming_loop(fake_stream_api, fake_stream_api, requests_queue, response_queues)

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
    fake_stream_api.request_timeout = 1
    fake_stream_api.decode_request = MagicMock(side_effect=lambda x: x["prompt"])
    fake_stream_api.batch = MagicMock(side_effect=lambda inputs: inputs)
    fake_stream_api.predict = MagicMock(side_effect=fake_predict)
    fake_stream_api.encode_response = MagicMock(side_effect=fake_encode)
    fake_stream_api.unbatch = MagicMock(side_effect=lambda inputs: inputs)
    fake_stream_api.format_encoded_response = MagicMock(side_effect=lambda x: x)

    requests_queue = Queue()
    requests_queue.put((0, "UUID-001", time.monotonic(), {"prompt": "Hello"}))
    requests_queue.put((0, "UUID-002", time.monotonic(), {"prompt": "World"}))
    response_queues = [FakeBatchStreamResponseQueue(num_streamed_outputs)]

    with pytest.raises(StopIteration, match="finish streaming"):
        run_batched_streaming_loop(
            fake_stream_api, fake_stream_api, requests_queue, response_queues, max_batch_size=2, batch_timeout=2
        )
    fake_stream_api.predict.assert_called_once_with(["Hello", "World"])
    fake_stream_api.encode_response.assert_called_once()


@patch("litserve.loops.run_batched_loop")
@patch("litserve.loops.run_single_loop")
def test_inference_worker(mock_single_loop, mock_batched_loop):
    inference_worker(*[MagicMock()] * 6, max_batch_size=2, batch_timeout=0, stream=False)
    mock_batched_loop.assert_called_once()

    inference_worker(*[MagicMock()] * 6, max_batch_size=1, batch_timeout=0, stream=False)
    mock_single_loop.assert_called_once()
