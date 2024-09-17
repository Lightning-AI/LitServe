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
import numpy as np

from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from litserve.loops import (
    run_single_loop,
    run_streaming_loop,
    run_batched_streaming_loop,
    inference_worker,
    run_batched_loop,
    run_single_preprocess_loop,
    run_batched_preprocess_loop,
)
from litserve.test_examples.openai_spec_example import OpenAIBatchingWithUsage
from litserve.utils import LitAPIStatus
import litserve as ls


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
        run_single_loop(lit_api_mock, None, requests_queue, None, response_queues)


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
        run_streaming_loop(fake_stream_api, fake_stream_api, requests_queue, None, response_queues)

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
            fake_stream_api, fake_stream_api, requests_queue, None, response_queues, max_batch_size=2, batch_timeout=2
        )
    fake_stream_api.predict.assert_called_once_with(["Hello", "World"])
    fake_stream_api.encode_response.assert_called_once()


@patch("litserve.loops.run_batched_loop")
@patch("litserve.loops.run_single_loop")
def test_inference_worker(mock_single_loop, mock_batched_loop):
    inference_worker(*[MagicMock()] * 7, max_batch_size=2, batch_timeout=0, stream=False)
    mock_batched_loop.assert_called_once()

    inference_worker(*[MagicMock()] * 7, max_batch_size=1, batch_timeout=0, stream=False)
    mock_single_loop.assert_called_once()


def test_run_single_loop():
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(target=run_single_loop, args=(lit_api, None, request_queue, None, response_queues))
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
    loop_thread = threading.Thread(target=run_single_loop, args=(lit_api, None, request_queue, None, response_queues))
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
        target=run_batched_loop, args=(lit_api, None, request_queue, None, response_queues, 2, 1)
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
        target=run_batched_loop, args=(lit_api, None, request_queue, None, response_queues, 2, 0.001)
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


def test_run_single_loop_with_preprocess():
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1
    lit_api.preprocess = lambda x: x + 1
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"

    request_queue = Queue()
    preprocess_queue = Queue()
    response_queues = [Queue()]

    # Put a request in the queue
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))

    # Run the preprocess loop in a separate thread
    import threading

    loop_thread = threading.Thread(
        target=run_single_preprocess_loop,
        args=(lit_api, None, request_queue, preprocess_queue, response_queues),
    )
    loop_thread.start()

    # Allow some time for processing
    time.sleep(0.5)

    # Check if the preprocessed data is in the preprocess_queue
    assert not preprocess_queue.empty()
    response_queue_id, uid, preprocessed_data = preprocess_queue.get()
    assert response_queue_id == 0
    assert uid == "UUID-001"
    assert preprocessed_data == 5.0

    # Stop the loop
    request_queue.put((None, None, None, None))
    loop_thread.join()


def test_run_batched_loop_with_preprocess():
    lit_api = ls.test_examples.SimpleBatchedAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1
    lit_api.preprocess = lambda batch: [x + 1 for x in batch]
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"

    request_queue = Queue()
    preprocess_queue = Queue()
    response_queues = [Queue()]

    # Put multiple requests in the queue
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))
    request_queue.put((0, "UUID-002", time.monotonic(), {"input": 5.0}))

    # Run the batched preprocess loop in a separate thread
    import threading

    loop_thread = threading.Thread(
        target=run_batched_preprocess_loop,
        args=(lit_api, None, request_queue, preprocess_queue, response_queues, 2, 0.1),
    )
    loop_thread.start()

    # Allow some time for processing
    time.sleep(0.5)

    # Check if the preprocessed batch is in the preprocess_queue
    assert not preprocess_queue.empty()
    result = preprocess_queue.get()
    assert isinstance(result, list), "Expected a list result from batched preprocessing"
    assert len(result) == 3, "Expected a list with 3 elements: [response_queue_ids, uids, preprocessed_batch]"

    response_queue_ids, uids, preprocessed_batch = result
    assert response_queue_ids == (0, 0), "Expected response_queue_ids to be (0, 0)"
    assert uids == ("UUID-001", "UUID-002"), "Expected uids to be ('UUID-001', 'UUID-002')"
    assert preprocessed_batch == [
        5.0,
        6.0,
    ], "Expected preprocessed batch to be [{'input': 5.0}, {'input': 6.0}]"

    # Stop the loop
    request_queue.put((None, None, None, None))
    loop_thread.join()


def test_run_single_loop_with_preprocess_timeout(caplog):
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 0.0001  # Very small timeout to force a timeout
    lit_api.preprocess = lambda x: x + 1
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"

    request_queue = Queue()
    preprocess_queue = Queue()
    response_queues = [Queue()]

    # Simulate an old request to trigger timeout
    request = (0, "UUID-001", time.monotonic() - 1, {"input": 4.0})
    request_queue.put(request)

    # Run the preprocess loop in a separate thread
    loop_thread = threading.Thread(
        target=run_single_preprocess_loop,
        args=(lit_api, None, request_queue, preprocess_queue, response_queues),
    )
    loop_thread.start()

    # Allow some time for processing
    time.sleep(0.5)

    # Stop the loop
    request_queue.put((None, None, None, None))
    loop_thread.join()

    # Check if the response queue has an error
    assert not response_queues[0].empty()
    uid, (response, status) = response_queues[0].get()
    assert uid == "UUID-001"
    assert isinstance(response, HTTPException), "Timeout should return an HTTPException"
    assert "Request UUID-001 was waiting in the queue for too long" in caplog.text


def test_run_batched_loop_with_preprocess_timeout(caplog):
    lit_api = ls.test_examples.SimpleBatchedAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 0.0001  # Very small timeout to force a timeout
    lit_api.preprocess = lambda batch: [x + 1 for x in batch]
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"

    request_queue = Queue()
    preprocess_queue = Queue()
    response_queues = [Queue()]

    # Simulate an old request to trigger timeout
    request = (0, "UUID-001", time.monotonic() - 1, {"input": 4.0})
    request_queue.put(request)

    # Run the batched preprocess loop in a separate thread
    loop_thread = threading.Thread(
        target=run_batched_preprocess_loop,
        args=(lit_api, None, request_queue, preprocess_queue, response_queues, 2, 0.1),
    )
    loop_thread.start()

    # Allow some time for processing
    time.sleep(0.5)

    # Stop the loop
    request_queue.put((None, None, None, None))
    loop_thread.join()

    # Check if the response queue has an error
    assert not response_queues[0].empty()
    uid, (response, status) = response_queues[0].get()
    assert uid == "UUID-001"
    assert isinstance(response, HTTPException), "Timeout should return an HTTPException"
    assert "Request UUID-001 was waiting in the queue for too long" in caplog.text


def test_run_single_loop_with_preprocess_inference_flow():
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1
    lit_api.preprocess = lambda x: x + 1
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"

    preprocess_queue = Queue()
    response_queue = Queue()
    response_queues = [response_queue]

    # Prepare the preprocessed data
    response_queue_id = 0
    uid = "UUID-001"
    preprocessed_data = 5.0  # Since preprocess adds 1 to the input 4.0
    preprocess_queue.put((response_queue_id, uid, preprocessed_data))

    # Run the inference loop in a separate daemon thread
    inference_thread = threading.Thread(
        target=run_single_loop, args=(lit_api, None, None, preprocess_queue, response_queues), daemon=True
    )
    inference_thread.start()

    # Wait for the response
    uid_received, (response, status) = response_queue.get(timeout=10)
    assert uid_received == "UUID-001"
    assert response == {"output": 25.0}, f"Expected output to be 25.0 but got {response}"

    # Allow the thread to exit
    time.sleep(0.1)


def test_run_batched_loop_with_preprocess_inference_flow():
    lit_api = ls.test_examples.SimpleBatchedAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1
    lit_api.preprocess = lambda batch: [x + 1 for x in batch]
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"

    preprocess_queue = Queue()
    response_queue = Queue()
    response_queues = [response_queue]

    # Prepare the preprocessed data
    response_queue_ids = (0, 0)
    uids = ("UUID-001", "UUID-002")
    preprocessed_batch = [5.0, 6.0]  # Inputs 4.0 +1 and 5.0 +1

    # Since lit_api decode request convert requests into np array
    preprocessed_batch = np.asarray(preprocessed_batch)
    preprocess_queue.put([response_queue_ids, uids, preprocessed_batch])

    # Run the inference loop in a separate daemon thread
    inference_thread = threading.Thread(
        target=run_batched_loop, args=(lit_api, None, None, preprocess_queue, response_queues, 2, 0.1), daemon=True
    )
    inference_thread.start()

    # Collect the responses
    outputs = {}
    for _ in range(2):
        uid_received, (response, status) = response_queue.get(timeout=10)
        outputs[uid_received] = response

    expected_output1 = {"output": 25.0}  # (5.0)^2
    expected_output2 = {"output": 36.0}  # (6.0)^2

    assert outputs["UUID-001"] == expected_output1, f"Expected {expected_output1} but got {outputs['UUID-001']}"
    assert outputs["UUID-002"] == expected_output2, f"Expected {expected_output2} but got {outputs['UUID-002']}"

    # Allow the thread to exit
    time.sleep(0.1)


def test_run_single_streaming_loop_with_preprocess_inference_flow():
    lit_api = ls.test_examples.SimpleStreamAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1
    lit_api.preprocess = lambda x: x
    lit_api.format_encoded_response = lambda y_enc: y_enc
    lit_api._sanitize(2, None)
    assert lit_api.model is not None, "Setup must initialize the model"

    preprocess_queue = Queue()
    response_queue = Queue()
    response_queues = [response_queue]

    # Prepare the preprocessed data
    response_queue_id = 0
    uid = "UUID-001"
    preprocessed_data = "Hello"
    preprocess_queue.put((response_queue_id, uid, preprocessed_data))

    # Run the streaming inference loop in a separate daemon thread
    inference_thread = threading.Thread(
        target=run_streaming_loop, args=(lit_api, None, None, preprocess_queue, response_queues), daemon=True
    )
    inference_thread.start()

    # Collect the streamed responses
    streamed_outputs = []
    while True:
        uid_received, (response, status) = response_queue.get(timeout=10)

        if status == LitAPIStatus.FINISH_STREAMING:
            break
        streamed_outputs.append(response)

    expected_outputs = [{"output": f"{i}: Hello"} for i in range(3)]

    assert streamed_outputs == expected_outputs, f"Expected {expected_outputs} but got {streamed_outputs}"

    # Allow the thread to exit
    time.sleep(0.1)


def test_run_streaming_loop():
    lit_api = ls.test_examples.SimpleStreamAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": "Hello"}))
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_streaming_loop, args=(lit_api, None, request_queue, None, response_queues)
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

    # Run the loop in a separate thread to allow it to be stopped
    loop_thread = threading.Thread(
        target=run_streaming_loop, args=(lit_api, None, request_queue, None, response_queues)
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
        target=run_batched_streaming_loop, args=(lit_api, spec, request_queue, None, response_queues, 2, 0.1)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put((None, None, None, None))
    loop_thread.join()

    response = response_queues[0].get(timeout=5)[1]
    assert response[0] == {"role": "assistant", "content": "10 + 6 is equal to 16."}
