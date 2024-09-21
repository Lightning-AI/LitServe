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
import json
import threading

import time
from queue import Queue

from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from litserve.callbacks import CallbackRunner
from litserve.loops import (
    run_single_loop,
    run_streaming_loop,
    run_batched_streaming_loop,
    inference_worker,
    run_batched_loop,
)
from litserve.test_examples.openai_spec_example import OpenAIBatchingWithUsage
from litserve.utils import LitAPIStatus
import litserve as ls

NOOP_CB_RUNNER = CallbackRunner()


@pytest.fixture
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
        run_single_loop(lit_api_mock, None, requests_queue, response_queues, callback_runner=NOOP_CB_RUNNER)


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
    request_evicted_status = {}

    with pytest.raises(StopIteration, match="exit loop"):
        run_streaming_loop(
            fake_stream_api,
            fake_stream_api,
            requests_queue,
            response_queues,
            request_evicted_status,
            callback_runner=NOOP_CB_RUNNER,
        )

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
            fake_stream_api,
            fake_stream_api,
            requests_queue,
            response_queues,
            max_batch_size=2,
            batch_timeout=2,
            callback_runner=NOOP_CB_RUNNER,
        )
    fake_stream_api.predict.assert_called_once_with(["Hello", "World"])
    fake_stream_api.encode_response.assert_called_once()


@patch("litserve.loops.run_batched_loop")
@patch("litserve.loops.run_single_loop")
def test_inference_worker(mock_single_loop, mock_batched_loop):
    inference_worker(
        *[MagicMock()] * 6,
        max_batch_size=2,
        batch_timeout=0,
        stream=False,
        workers_setup_status={},
        request_evicted_status={},
        callback_runner=NOOP_CB_RUNNER,
    )
    mock_batched_loop.assert_called_once()

    inference_worker(
        *[MagicMock()] * 6,
        max_batch_size=1,
        batch_timeout=0,
        stream=False,
        workers_setup_status={},
        request_evicted_status={},
        callback_runner=NOOP_CB_RUNNER,
    )
    mock_single_loop.assert_called_once()


def test_run_single_loop():
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_single_loop, args=(lit_api, None, request_queue, response_queues, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put((None, None, None, None))
    loop_thread.join()

    response = response_queues[0].get()
    assert response == ("UUID-001", ({"output": 16.0}, LitAPIStatus.OK))


def test_run_single_loop_timeout(caplog):
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 0.0001

    request_queue = Queue()
    request = (0, "UUID-001", time.monotonic(), {"input": 4.0})
    time.sleep(0.1)
    request_queue.put(request)
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_single_loop, args=(lit_api, None, request_queue, response_queues, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    request_queue.put((None, None, None, None))
    loop_thread.join()
    assert "Request UUID-001 was waiting in the queue for too long" in caplog.text
    assert isinstance(response_queues[0].get()[1][0], HTTPException), "Timeout should return an HTTPException"


def test_run_batched_loop():
    lit_api = ls.test_examples.SimpleBatchedAPI()
    lit_api.setup(None)
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"
    lit_api.request_timeout = 1

    request_queue = Queue()
    # response_queue_id, uid, timestamp, x_enc
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))
    request_queue.put((0, "UUID-002", time.monotonic(), {"input": 5.0}))
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_batched_loop, args=(lit_api, None, request_queue, response_queues, 2, 1, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put((None, None, None, None))
    loop_thread.join()

    response_1 = response_queues[0].get(timeout=10)
    response_2 = response_queues[0].get(timeout=10)
    assert response_1 == ("UUID-001", ({"output": 16.0}, LitAPIStatus.OK))
    assert response_2 == ("UUID-002", ({"output": 25.0}, LitAPIStatus.OK))


def test_run_batched_loop_timeout(caplog):
    lit_api = ls.test_examples.SimpleBatchedAPI()
    lit_api.setup(None)
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"
    lit_api.request_timeout = 0.1

    request_queue = Queue()
    # response_queue_id, uid, timestamp, x_enc
    r1 = (0, "UUID-001", time.monotonic(), {"input": 4.0})
    time.sleep(0.1)
    request_queue.put(r1)
    r2 = (0, "UUID-002", time.monotonic(), {"input": 5.0})
    request_queue.put(r2)
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_batched_loop, args=(lit_api, None, request_queue, response_queues, 2, 0.001, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    assert "Request UUID-001 was waiting in the queue for too long" in caplog.text
    resp1 = response_queues[0].get(timeout=10)[1]
    resp2 = response_queues[0].get(timeout=10)[1]
    assert isinstance(resp1[0], HTTPException), "First request was timed out"
    assert resp2[0] == {"output": 25.0}, "Second request wasn't timed out"

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put((None, None, None, None))
    loop_thread.join()


def test_run_streaming_loop():
    lit_api = ls.test_examples.SimpleStreamAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": "Hello"}))
    response_queues = [Queue()]
    request_evicted_status = {}

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_streaming_loop,
        args=(lit_api, None, request_queue, response_queues, request_evicted_status, NOOP_CB_RUNNER),
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put((None, None, None, None))
    loop_thread.join()

    for i in range(3):
        response = response_queues[0].get(timeout=10)
        response = json.loads(response[1][0])
        assert response == {"output": f"{i}: Hello"}


def test_run_streaming_loop_timeout(caplog):
    lit_api = ls.test_examples.SimpleStreamAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 0.1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic() - 5, {"input": "Hello"}))
    response_queues = [Queue()]
    request_evicted_status = {}

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_streaming_loop,
        args=(lit_api, None, request_queue, response_queues, request_evicted_status, NOOP_CB_RUNNER),
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put((None, None, None, None))
    loop_thread.join()

    assert "Request UUID-001 was waiting in the queue for too long" in caplog.text
    response = response_queues[0].get(timeout=10)[1]
    assert isinstance(response[0], HTTPException), "request was timed out"


def off_test_run_batched_streaming_loop(openai_request_data):
    lit_api = OpenAIBatchingWithUsage()
    lit_api.setup(None)
    lit_api.request_timeout = 1
    lit_api.stream = True
    spec = ls.OpenAISpec()
    lit_api._sanitize(2, spec)

    request_queue = Queue()
    # response_queue_id, uid, timestamp, x_enc
    r1 = (0, "UUID-001", time.monotonic(), openai_request_data)
    r2 = (0, "UUID-002", time.monotonic(), openai_request_data)
    request_queue.put(r1)
    request_queue.put(r2)
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_batched_streaming_loop, args=(lit_api, spec, request_queue, response_queues, 2, 0.1, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put((None, None, None, None))
    loop_thread.join()

    response = response_queues[0].get(timeout=5)[1]
    assert response[0] == {"role": "assistant", "content": "10 + 6 is equal to 16."}
