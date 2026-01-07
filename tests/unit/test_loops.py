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
import contextlib
import inspect
import io
import json
import re
import sys
import threading
import time
from collections.abc import AsyncGenerator
from queue import Empty, Queue
from typing import Optional
from unittest.mock import ANY, MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve import LitAPI
from litserve.callbacks import CallbackRunner
from litserve.loops import BatchedStreamingLoop, LitLoop, Output, StreamingLoop, inference_worker
from litserve.loops.base import (
    _SENTINEL_VALUE,
    DefaultLoop,
    _async_inject_context,
    _handle_async_function,
    _sync_fn_to_async_fn,
)
from litserve.loops.continuous_batching_loop import (
    ContinuousBatchingLoop,
    notify_timed_out_requests,
)
from litserve.loops.simple_loops import BatchedLoop, SingleLoop
from litserve.specs.base import LitSpec
from litserve.test_examples.openai_spec_example import OpenAIBatchingWithUsage
from litserve.transport import MPQueueTransport
from litserve.transport.base import MessageTransport
from litserve.utils import LitAPIStatus, LoopResponseType, wrap_litserve_start

NOOP_CB_RUNNER = CallbackRunner()


class MockMPQueueTransport(MPQueueTransport):
    def __init__(self, num_consumers=1):
        self._closed = False
        self._mp_terminate_event = None
        self._queues = [Queue() for _ in range(num_consumers)]


@pytest.fixture
def mock_transport():
    return MockMPQueueTransport()


@pytest.fixture
def loop_args():
    requests_queue = Queue()
    requests_queue.put((0, "uuid-123", time.monotonic(), 1))  # response_queue_id, uid, timestamp, x_enc
    requests_queue.put((1, "uuid-234", time.monotonic(), 2))

    lit_api_mock = MagicMock()
    lit_api_mock.request_timeout = 1
    lit_api_mock.decode_request = MagicMock(side_effect=lambda x: x["input"])
    return lit_api_mock, requests_queue


class TestQueue(Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sentinel_seen = False

    def get(self, timeout=None):
        if self._sentinel_seen:
            raise Empty  # Simulate queue being empty after sentinel
        item = super().get(timeout=timeout)
        # Sentinel: (None, None, None, None)
        if item == _SENTINEL_VALUE:
            self._sentinel_seen = True
            raise KeyboardInterrupt  # Triggers loop exit in your code
        return item


class AsyncTestLitAPI(LitAPI):
    def setup(self, device):
        pass

    async def decode_request(self, request):
        return request["input"]

    async def predict(self, x):
        return x**2

    async def encode_response(self, output):
        return {"output": output}


@pytest.fixture
def async_loop_args():
    requests_queue = TestQueue()
    requests_queue.put((0, "uuid-123", time.monotonic(), {"input": 1}))
    requests_queue.put((1, "uuid-234", time.monotonic(), {"input": 2}))
    requests_queue.put(_SENTINEL_VALUE)

    lit_api = AsyncTestLitAPI()
    return lit_api, requests_queue


class DummyMessageTransport(MessageTransport):
    def send(self, item, consumer_id, block=True, timeout=None):
        raise StopIteration("exit loop")

    async def areceive(self, timeout: Optional[int] = None, consumer_id: Optional[int] = None) -> dict:
        pass


def test_single_loop(loop_args):
    lit_api_mock, requests_queue = loop_args
    lit_api_mock.unbatch.side_effect = None
    transport = DummyMessageTransport()

    lit_loop = SingleLoop()
    with pytest.raises(StopIteration, match="exit loop"):
        lit_loop.run_single_loop(lit_api_mock, requests_queue, transport, callback_runner=NOOP_CB_RUNNER)


@pytest.mark.asyncio
async def test_single_loop_process_single_async_request(async_loop_args, mock_transport):
    lit_api_mock, requests_queue = async_loop_args

    # Get a request from the queue (already populated by the fixture)
    request = requests_queue.get()
    loop = SingleLoop()
    await loop._process_single_request(
        request,
        lit_api_mock,
        mock_transport,
        NOOP_CB_RUNNER,
    )
    response = await mock_transport.areceive(consumer_id=request[0])
    expected_output = request[3]["input"] ** 2
    assert response == (
        request[1],
        ({"output": expected_output}, ls.utils.LitAPIStatus.OK, ls.utils.LoopResponseType.REGULAR, ANY),
    )


@pytest.mark.skipif(
    sys.platform == "linux" and sys.version_info[:2] == (3, 12),
    reason="Event loop handling issue on Ubuntu Python 3.12"
)
def test_run_single_loop_with_async(async_loop_args, monkeypatch):
    mock_transport = MockMPQueueTransport(num_consumers=2)
    lit_api_mock, requests_queue = async_loop_args

    loop = SingleLoop()
    loop._restart_workers = True

    # Patch kill to do nothing in test
    monkeypatch.setattr(loop, "kill", lambda: None)

    # Expected to break the loop in test
    with contextlib.suppress(KeyboardInterrupt):
        loop._run_single_loop_with_async(lit_api_mock, requests_queue, mock_transport, NOOP_CB_RUNNER)

    response = asyncio.get_event_loop().run_until_complete(mock_transport.areceive(consumer_id=0))
    assert response == ("uuid-123", ((), "START", ls.utils.LoopResponseType.REGULAR, ANY))
    response = asyncio.get_event_loop().run_until_complete(mock_transport.areceive(consumer_id=0))
    assert response == ("uuid-123", ({"output": 1}, ls.utils.LitAPIStatus.OK, ls.utils.LoopResponseType.REGULAR, ANY))
    response = asyncio.get_event_loop().run_until_complete(mock_transport.areceive(consumer_id=1))
    assert response == ("uuid-234", ((), "START", ls.utils.LoopResponseType.REGULAR, ANY))
    response = asyncio.get_event_loop().run_until_complete(mock_transport.areceive(consumer_id=1))
    assert response == ("uuid-234", ({"output": 4}, ls.utils.LitAPIStatus.OK, ls.utils.LoopResponseType.REGULAR, ANY))


class FakeStreamSender(DummyMessageTransport):
    def __init__(self, num_streamed_outputs):
        super().__init__()
        self.num_streamed_outputs = num_streamed_outputs
        self.count = 0

    def send(self, item, consumer_id, block=False, timeout=None):
        uid, args = item
        response, status, response_type, worker_id = args
        if status == LitAPIStatus.START:
            return

        if self.count >= self.num_streamed_outputs:
            raise StopIteration("exit loop")
        assert response == f"{self.count}", "This streaming loop generates number from 0 to 9 which is sent via Queue"
        assert response_type == LoopResponseType.STREAMING, "Streaming loop must return streaming response"
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
    transport = FakeStreamSender(num_streamed_outputs)

    lit_loop = StreamingLoop()
    with pytest.raises(StopIteration, match="exit loop"):
        lit_loop.run_streaming_loop(
            fake_stream_api,
            requests_queue,
            transport,
            callback_runner=NOOP_CB_RUNNER,
        )

    fake_stream_api.predict.assert_called_once_with("Hello")
    fake_stream_api.encode_response.assert_called_once()


class AsyncTestStreamLitAPI(LitAPI):
    def setup(self, device) -> None:
        pass

    async def decode_request(self, request):
        return request["input"]

    async def predict(self, x):
        for i in range(x):
            yield {"output": i}

    async def encode_response(self, output):
        async for out in output:
            yield out["output"]


@pytest.mark.asyncio
async def test_streaming_loop_process_streaming_request(mock_transport):
    request = (0, "UUID-1234", time.monotonic(), {"input": 5})

    lit_api = AsyncTestStreamLitAPI()
    loop = StreamingLoop()
    await loop._process_streaming_request(
        request,
        lit_api,
        mock_transport,
        NOOP_CB_RUNNER,
    )

    for i in range(5):
        response = await mock_transport.areceive(consumer_id=request[0])
        assert response == (
            request[1],
            (i, ls.utils.LitAPIStatus.OK, ls.utils.LoopResponseType.STREAMING, ANY),
        )


@pytest.mark.skipif(
    sys.platform == "linux" and sys.version_info[:2] == (3, 12),
    reason="Event loop handling issue on Ubuntu Python 3.12"
)
def test_run_streaming_loop_with_async(mock_transport, monkeypatch):
    requests_queue = TestQueue()
    requests_queue.put((0, "uuid-123", time.monotonic(), {"input": 5}))
    requests_queue.put(_SENTINEL_VALUE)  # Sentinel to stop the loop

    lit_api = AsyncTestStreamLitAPI()
    loop = StreamingLoop()
    loop._restart_workers = True

    # Patch kill to do nothing in test
    monkeypatch.setattr(loop, "kill", lambda: None)

    # Expected to break the loop in test
    with contextlib.suppress(KeyboardInterrupt):
        loop.run_streaming_loop_async(lit_api, requests_queue, mock_transport, NOOP_CB_RUNNER)

    for i in range(6):
        response = asyncio.get_event_loop().run_until_complete(mock_transport.areceive(consumer_id=0))
        if i == 0:
            assert response == (
                "uuid-123",
                (((), "START", ls.utils.LoopResponseType.STREAMING, ANY)),
            )
        else:
            assert response == (
                "uuid-123",
                (i - 1, ls.utils.LitAPIStatus.OK, ls.utils.LoopResponseType.STREAMING, ANY),
            )


class FakeBatchStreamTransport(DummyMessageTransport):
    def __init__(self, num_streamed_outputs):
        super().__init__()
        self.num_streamed_outputs = num_streamed_outputs
        self.count = 0

    def send(self, item, consumer_id=0, block=False, timeout=None):
        uid, args = item
        response, status, response_type, worker_id = args
        if status == LitAPIStatus.START:
            return

        if status == LitAPIStatus.FINISH_STREAMING:
            raise StopIteration("interrupt iteration")
        if status == LitAPIStatus.ERROR:
            assert self.count // 2 == self.num_streamed_outputs, (
                f"Loop count must have incremented for {self.num_streamed_outputs} times."
            )
            raise StopIteration("finish streaming")

        assert response == f"{self.count // 2}", (
            f"streaming loop generates number from 0 to 9 which is sent via Queue. {args}, count:{self.count}"
        )
        assert response_type == LoopResponseType.STREAMING, "Streaming loop must return streaming response"
        self.count += 1


def test_batched_streaming_loop(mock_transport):
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
    fake_stream_api.max_batch_size = 2
    fake_stream_api.batch_timeout = 2

    requests_queue = Queue()
    requests_queue.put((0, "UUID-001", time.monotonic(), {"prompt": "Hello"}))
    requests_queue.put((0, "UUID-002", time.monotonic(), {"prompt": "World"}))

    lit_loop = BatchedStreamingLoop()
    transport = FakeBatchStreamTransport(num_streamed_outputs)
    with pytest.raises(StopIteration, match="finish streaming"):
        lit_loop.run_batched_streaming_loop(
            fake_stream_api,
            requests_queue,
            transport=transport,
            callback_runner=NOOP_CB_RUNNER,
        )
    fake_stream_api.predict.assert_called_once_with(["Hello", "World"])
    fake_stream_api.encode_response.assert_called_once()


@patch("litserve.loops.simple_loops.BatchedLoop.run_batched_loop")
@patch("litserve.loops.simple_loops.SingleLoop.run_single_loop")
def test_inference_worker(mock_single_loop, mock_batched_loop):
    lit_api_mock = MagicMock()
    lit_api_mock.max_batch_size = 2
    lit_api_mock.batch_timeout = 0
    lit_api_mock.enable_async = False
    lit_api_mock.stream = False
    lit_api_mock.api_path = "/predict"
    lit_api_mock.loop = "auto"

    inference_worker(
        lit_api_mock,
        "cpu",
        0,
        MagicMock(),
        MagicMock(),
        workers_setup_status={},
        callback_runner=NOOP_CB_RUNNER,
        restart_workers=True,
    )
    mock_batched_loop.assert_called_once()

    lit_api_mock = MagicMock()
    lit_api_mock.max_batch_size = 1
    lit_api_mock.batch_timeout = 0
    lit_api_mock.enable_async = False
    lit_api_mock.stream = False
    lit_api_mock.api_path = "/predict"
    lit_api_mock.loop = "auto"

    inference_worker(
        lit_api_mock,
        "cpu",
        0,
        MagicMock(),
        MagicMock(),
        workers_setup_status={},
        callback_runner=NOOP_CB_RUNNER,
        restart_workers=True,
    )
    mock_single_loop.assert_called_once()


@pytest.mark.asyncio
async def test_run_single_loop(mock_transport):
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))
    transport = mock_transport

    # Run the loop in a separate thread to allow it to be stopped
    lit_loop = SingleLoop()
    lit_loop._restart_workers = True
    loop_thread = threading.Thread(
        target=lit_loop.run_single_loop, args=(lit_api, request_queue, transport, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put(_SENTINEL_VALUE)
    loop_thread.join()

    response = await transport.areceive(consumer_id=0)
    response = await transport.areceive(consumer_id=0)
    assert response == ("UUID-001", ({"output": 16.0}, LitAPIStatus.OK, LoopResponseType.REGULAR, ANY))


@pytest.mark.asyncio
async def test_run_single_loop_timeout():
    stream = io.StringIO()
    ls.configure_logging(stream=stream)

    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 0.0001

    request_queue = Queue()
    transport = MockMPQueueTransport()
    old_request = (0, "UUID-001", time.monotonic(), {"input": 4.0})
    time.sleep(0.1)  # Age the request
    request_queue.put(old_request)

    lit_loop = SingleLoop()
    lit_loop._restart_workers = True
    loop_thread = threading.Thread(
        target=lit_loop.run_single_loop, args=(lit_api, request_queue, transport, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    _, (response, status, _, _) = await transport.areceive(consumer_id=0)
    _, (response, status, _, _) = await transport.areceive(consumer_id=0)
    assert isinstance(response, HTTPException)
    assert response.status_code == 504
    assert "Request UUID-001 was waiting in the queue for too long" in stream.getvalue()

    request_queue.put(_SENTINEL_VALUE)
    loop_thread.join()


@pytest.mark.asyncio
async def test_run_batched_loop():
    lit_api = ls.test_examples.SimpleBatchedAPI()
    lit_api.setup(None)
    lit_api.max_batch_size = 2
    lit_api.batch_timeout = 1
    lit_api.request_timeout = 1
    lit_api.pre_setup(spec=None)

    request_queue = Queue()
    transport = MockMPQueueTransport(1)

    requests = [(0, "UUID-001", time.monotonic(), {"input": 4.0}), (0, "UUID-002", time.monotonic(), {"input": 5.0})]
    for req in requests:
        request_queue.put(req)

    lit_loop = BatchedLoop()
    lit_loop._restart_workers = True
    loop_thread = threading.Thread(
        target=lit_loop.run_batched_loop,
        args=(lit_api, request_queue, transport, NOOP_CB_RUNNER),
    )
    loop_thread.start()

    expected_responses = [
        ("UUID-001", ({"output": 16.0}, LitAPIStatus.OK, LoopResponseType.REGULAR, ANY)),
        ("UUID-002", ({"output": 25.0}, LitAPIStatus.OK, LoopResponseType.REGULAR, ANY)),
    ]

    await transport.areceive(0, timeout=10)
    await transport.areceive(0, timeout=10)

    for expected in expected_responses:
        actual = await transport.areceive(0, timeout=10)
        assert actual == expected, f"Expected {expected}, got {actual}"

    request_queue.put(_SENTINEL_VALUE)
    loop_thread.join()


@pytest.mark.asyncio
async def test_run_batched_loop_timeout(mock_transport):
    stream = io.StringIO()
    ls.configure_logging(stream=stream)

    lit_api = ls.test_examples.SimpleBatchedAPI()
    lit_api.setup(None)
    lit_api.max_batch_size = 2
    lit_api.batch_timeout = 0.001
    lit_api.request_timeout = 0.1
    lit_api.pre_setup(spec=None)

    request_queue = Queue()
    transport = mock_transport

    # First request will time out, second will succeed
    requests = [
        (0, "UUID-001", time.monotonic() - 0.2, {"input": 4.0}),  # Old request
        (0, "UUID-002", time.monotonic(), {"input": 5.0}),  # Fresh request
    ]
    for req in requests:
        request_queue.put(req)

    lit_loop = BatchedLoop()
    lit_loop._restart_workers = True
    loop_thread = threading.Thread(
        target=lit_loop.run_batched_loop,
        args=(lit_api, request_queue, transport, NOOP_CB_RUNNER),
    )
    loop_thread.start()

    await transport.areceive(0, timeout=10)
    await transport.areceive(0, timeout=10)

    # First response should be timeout error
    _, (response1, _, _, _) = await transport.areceive(0, timeout=10)
    assert isinstance(response1, HTTPException)
    assert "Request UUID-001 was waiting in the queue for too long" in stream.getvalue()

    # Second response should succeed
    _, (response2, _, _, _) = await transport.areceive(consumer_id=0, timeout=10)
    assert response2 == {"output": 25.0}

    request_queue.put(_SENTINEL_VALUE)
    loop_thread.join()


@pytest.mark.asyncio
async def test_run_streaming_loop(mock_transport):
    lit_api = ls.test_examples.SimpleStreamAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": "Hello"}))

    # Run the loop in a separate thread to allow it to be stopped
    lit_loop = StreamingLoop()
    lit_loop._restart_workers = True
    loop_thread = threading.Thread(
        target=lit_loop.run_streaming_loop, args=(lit_api, request_queue, mock_transport, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put(_SENTINEL_VALUE)
    loop_thread.join()

    await mock_transport.areceive(0, timeout=10)

    for i in range(3):
        response = await mock_transport.areceive(0, timeout=10)
        response = json.loads(response[1][0])
        assert response == {"output": f"{i}: Hello"}


@pytest.mark.asyncio
async def test_run_streaming_loop_timeout(mock_transport):
    stream = io.StringIO()
    ls.configure_logging(stream=stream)
    lit_api = ls.test_examples.SimpleStreamAPI()
    lit_api.setup(None)
    lit_api.request_timeout = 0.1

    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic() - 5, {"input": "Hello"}))

    # Run the loop in a separate thread to allow it to be stopped
    lit_loop = StreamingLoop()
    lit_loop._restart_workers = True
    loop_thread = threading.Thread(
        target=lit_loop.run_streaming_loop, args=(lit_api, request_queue, mock_transport, NOOP_CB_RUNNER)
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put(_SENTINEL_VALUE)
    loop_thread.join()

    assert "Request UUID-001 was waiting in the queue for too long" in stream.getvalue()
    response = await mock_transport.areceive(0, timeout=10)
    response = await mock_transport.areceive(0, timeout=10)
    assert isinstance(response[1][0], HTTPException), "request was timed out"


def off_test_run_batched_streaming_loop(openai_request_data):
    lit_api = OpenAIBatchingWithUsage()
    lit_api.setup(None)
    lit_api.request_timeout = 1
    lit_api.stream = True
    lit_api.max_batch_size = 2
    lit_api.batch_timeout = 0.1
    spec = ls.OpenAISpec()
    lit_api.pre_setup(spec=spec, timeout=30)

    request_queue = Queue()
    # response_queue_id, uid, timestamp, x_enc
    r1 = (0, "UUID-001", time.monotonic(), openai_request_data)
    r2 = (0, "UUID-002", time.monotonic(), openai_request_data)
    request_queue.put(r1)
    request_queue.put(r2)
    response_queues = [Queue()]

    # Run the loop in a separate thread to allow it to be stopped
    lit_loop = BatchedStreamingLoop()
    lit_loop._restart_workers = True
    loop_thread = threading.Thread(
        target=lit_loop.run_batched_streaming_loop,
        args=(lit_api, spec, request_queue, response_queues, NOOP_CB_RUNNER),
    )
    loop_thread.start()

    # Allow some time for the loop to process
    time.sleep(1)

    # Stop the loop by putting a sentinel value in the queue
    request_queue.put(_SENTINEL_VALUE)
    loop_thread.join()

    response = response_queues[0].get(timeout=5)[1]
    assert response[0] == {"role": "assistant", "content": "10 + 6 is equal to 16."}


class TestLoop(LitLoop):
    def __call__(
        self,
        lit_api: LitAPI,
        device: str,
        worker_id: int,
        request_queue: Queue,
        transport: MessageTransport,
        workers_setup_status: dict[int, str],
        callback_runner: CallbackRunner,
    ):
        try:
            self.run(
                lit_api,
                device,
                worker_id,
                request_queue,
                transport,
                workers_setup_status,
                callback_runner,
            )
        except StopIteration as e:
            return e

    def run(
        self,
        lit_api: LitAPI,
        device: str,
        worker_id: int,
        request_queue: Queue,
        transport: MessageTransport,
        workers_setup_status: dict[int, str],
        callback_runner: CallbackRunner,
    ):
        item = request_queue.get()
        if item is None:
            return

        response_queue_id, uid, timestamp, x_enc = item
        cache = lit_api.load_cache(x_enc)
        x = lit_api.decode_request(x_enc) * cache
        response = lit_api.predict(x)
        response_enc = lit_api.encode_response(response)
        transport.send(
            (uid, (response_enc, LitAPIStatus.OK, LoopResponseType.REGULAR, ANY)), consumer_id=response_queue_id
        )
        raise StopIteration("exit loop")


@pytest.mark.asyncio
async def test_custom_loop(mock_transport):
    loop = TestLoop()
    lit_api = MagicMock(request_timeout=1)
    lit_api.load_cache = MagicMock(return_value=1.0)
    lit_api.encode_response = MagicMock(return_value={"output": 16.0})
    request_queue = Queue()
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))

    loop(lit_api, "cpu", 0, request_queue, mock_transport, {}, NOOP_CB_RUNNER)
    response = await mock_transport.areceive(0)
    assert response[0] == "UUID-001"
    assert response[1][0] == {"output": 16.0}
    lit_api.load_cache.assert_called_once()
    lit_api.load_cache.assert_called_with({"input": 4.0})


class TestLitAPI(ls.test_examples.SimpleLitAPI):
    def load_cache(self, x):
        return 10


@pytest.mark.asyncio
@pytest.mark.parametrize("fast_queue", [True, False])
async def test_loop_with_server_async(fast_queue):
    loop = TestLoop()
    loop._restart_workers = True
    lit_api = TestLitAPI()
    server = ls.LitServer(lit_api, loop=loop, fast_queue=fast_queue)

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            response = await ac.post("/predict", json={"input": 4.0}, timeout=5)
            assert response.json() == {"output": 1600.0}


def test_loop_with_server_sync():
    loop = TestLoop()
    loop._restart_workers = True
    lit_api = TestLitAPI()
    server = ls.LitServer(lit_api, loop=loop, fast_queue=True)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0}, timeout=5)
        assert response.json() == {"output": 1600.0}  # use LitAPI.load_cache to multiply the input by 10


def test_get_default_loop():
    lit_api = MagicMock()
    lit_api.stream = False
    lit_api.max_batch_size = 1
    loop = ls.loops.get_default_loop(lit_api.stream, lit_api.max_batch_size)
    loop._restart_workers = True
    assert isinstance(loop, ls.loops.SingleLoop), "SingleLoop must be returned when stream=False"

    lit_api = MagicMock()
    lit_api.stream = False
    lit_api.max_batch_size = 4
    loop = ls.loops.get_default_loop(lit_api.stream, lit_api.max_batch_size)
    loop._restart_workers = True
    assert isinstance(loop, ls.loops.BatchedLoop), "BatchedLoop must be returned when stream=False and max_batch_size>1"

    lit_api = MagicMock()
    lit_api.stream = True
    lit_api.max_batch_size = 1
    loop = ls.loops.get_default_loop(lit_api.stream, lit_api.max_batch_size)
    loop._restart_workers = True
    assert isinstance(loop, ls.loops.StreamingLoop), "StreamingLoop must be returned when stream=True"

    lit_api = MagicMock()
    lit_api.stream = True
    lit_api.max_batch_size = 4
    loop = ls.loops.get_default_loop(lit_api.stream, lit_api.max_batch_size)
    loop._restart_workers = True
    assert isinstance(loop, ls.loops.BatchedStreamingLoop), (
        "BatchedStreamingLoop must be returned when stream=True and max_batch_size>1"
    )


def test_get_default_loop_enable_async():
    lit_api = MagicMock()
    lit_api.max_batch_size = 2
    lit_api.enable_async = True
    with pytest.raises(
        ValueError, match="Async batching is not supported. Please use enable_async=False with batching."
    ):
        ls.loops.get_default_loop(lit_api.stream, lit_api.max_batch_size, lit_api.enable_async)


@pytest.fixture
def lit_loop_setup():
    lit_loop = LitLoop()
    lit_loop._restart_workers = True
    lit_api = MagicMock(request_timeout=0.1)
    request_queue = Queue()
    return lit_loop, lit_api, request_queue


def test_lit_loop_get_batch_requests(lit_loop_setup):
    lit_loop, lit_api, request_queue = lit_loop_setup
    lit_api.max_batch_size = 2
    lit_api.batch_timeout = 0.1
    request_queue.put((0, "UUID-001", time.monotonic(), {"input": 4.0}))
    request_queue.put((0, "UUID-002", time.monotonic(), {"input": 5.0}))
    batches, timed_out_uids = lit_loop.get_batch_requests(lit_api, request_queue, MagicMock())
    assert len(batches) == 2
    assert batches == [(0, "UUID-001", {"input": 4.0}), (0, "UUID-002", {"input": 5.0})]
    assert timed_out_uids == []


def test_lit_loop_get_request(lit_loop_setup):
    lit_loop, _, request_queue = lit_loop_setup
    t = time.monotonic()
    request_queue.put((0, "UUID-001", t, {"input": 4.0}))
    response_queue_id, uid, timestamp, x_enc = lit_loop.get_request(request_queue, timeout=1)
    assert uid == "UUID-001"
    assert response_queue_id == 0
    assert timestamp == t
    assert x_enc == {"input": 4.0}
    assert lit_loop.get_request(request_queue, timeout=0.001) is None


@pytest.mark.asyncio
async def test_lit_loop_put_response(lit_loop_setup, mock_transport):
    lit_loop, _, request_queue = lit_loop_setup
    lit_loop.put_response(mock_transport, 0, "UUID-001", {"output": 16.0}, LitAPIStatus.OK, LoopResponseType.REGULAR)
    response = await mock_transport.areceive(0)
    assert response == ("UUID-001", ({"output": 16.0}, LitAPIStatus.OK, LoopResponseType.REGULAR, ANY))


def test_notify_timed_out_requests():
    response_queues = [Queue()]

    # Simulate timed out requests
    timed_out_uids = [(0, "UUID-001"), (0, "UUID-002")]

    # Call the function to notify timed out requests
    notify_timed_out_requests(response_queues, timed_out_uids)

    # Check the responses in the response queue
    response_1 = response_queues[0].get()
    response_2 = response_queues[0].get()

    assert response_1[0] == "UUID-001"
    assert response_1[1][1] == LitAPIStatus.ERROR
    assert isinstance(response_1[1][0], HTTPException)
    assert response_2[0] == "UUID-002"
    assert isinstance(response_2[1][0], HTTPException)
    assert response_2[1][1] == LitAPIStatus.ERROR


class ContinuousBatchingAPI(ls.LitAPI):
    def setup(self, spec: Optional[LitSpec]):
        self.model = {}

    def add_request(self, uid: str, request):
        print(f"Adding to request_queue at {time.monotonic()}")
        self.model[uid] = {"outputs": list(range(5))}

    def decode_request(self, input: str):
        return input

    def encode_response(self, output: str):
        return {"output": output}

    def has_capacity(self) -> bool:
        return True

    def has_active_requests(self) -> bool:
        return bool(self.model)

    def step(self, prev_outputs: Optional[list[Output]]) -> list[Output]:
        outputs = []
        for k in self.model:
            v = self.model[k]
            if v["outputs"]:
                o = v["outputs"].pop(0)
                outputs.append(Output(k, o, LitAPIStatus.OK))
        keys = list(self.model.keys())
        for k in keys:
            if k not in [o.uid for o in outputs]:
                outputs.append(Output(k, "", LitAPIStatus.FINISH_STREAMING))
                del self.model[k]
        return outputs


@pytest.mark.parametrize(
    ("stream", "max_batch_size", "error_msg"),
    [
        (True, 4, "`lit_api.unbatch` must generate values using `yield`."),
        (True, 1, "`lit_api.encode_response` must generate values using `yield`."),
    ],
)
def test_default_loop_pre_setup_error(stream, max_batch_size, error_msg):
    lit_api = ls.test_examples.SimpleLitAPI()
    lit_api.stream = stream
    lit_api.max_batch_size = max_batch_size
    loop = DefaultLoop()
    with pytest.raises(ValueError, match=error_msg):
        loop.pre_setup(lit_api, None)


@pytest.fixture
def continuous_batching_setup(monkeypatch, mock_transport):
    lit_api = ContinuousBatchingAPI()
    lit_api.stream = True
    lit_api.request_timeout = 0.1
    lit_api.max_batch_size = 2
    lit_api.batch_timeout = 0.1
    lit_api.pre_setup(spec=None)
    lit_api.setup(None)
    request_queue = Queue()

    lit_loop = ContinuousBatchingLoop()
    return lit_api, lit_loop, request_queue, mock_transport


def test_continuous_batching_pre_setup(continuous_batching_setup):
    lit_api, lit_loop, request_queue, mock_transport = continuous_batching_setup
    lit_api.stream = False
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Continuous batching loop requires streaming to be enabled. Please set LitServe(..., stream=True)"
        ),
    ):
        lit_loop.pre_setup(lit_api, None)


@pytest.mark.asyncio
async def test_continuous_batching_run(continuous_batching_setup):
    lit_api, lit_loop, request_queue, mock_transport = continuous_batching_setup
    response_queue_id, uid, _, input = (0, "UUID-001", time.monotonic(), {"input": "Hello"})
    lit_loop.add_request(uid, input, lit_api, None)
    lit_loop.response_queue_ids[uid] = response_queue_id
    await lit_loop.run(lit_api, "cpu", 0, request_queue, mock_transport, {}, NOOP_CB_RUNNER)

    results = []
    for i in range(5):
        response = await mock_transport.areceive(0)
        uid, (response_data, status, response_type, _) = response
        o = json.loads(response_data)["output"]
        assert o == i
        assert status == LitAPIStatus.OK
        assert uid == "UUID-001"
        results.append(o)
    assert results == list(range(5)), "API must return a sequence of numbers from 0 to 4"
    response = await mock_transport.areceive(0)
    uid, (response_data, status, response_type, _) = response
    o = json.loads(response_data)["output"]
    assert o == ""
    assert status == LitAPIStatus.FINISH_STREAMING
    assert response_type == LoopResponseType.STREAMING


@pytest.mark.asyncio
async def test_handle_async_function():
    async def async_func():
        return "async"

    def sync_func():
        return "sync"

    async def async_gen():
        for i in range(3):
            yield i

    assert await _handle_async_function(async_func) == "async"
    assert await _handle_async_function(sync_func) == "sync"
    async_gen = await _handle_async_function(async_gen)
    assert isinstance(async_gen, AsyncGenerator)


@pytest.mark.asyncio
async def test_async_inject_context():
    async def async_func(x, context=0):
        return x * context["a"]

    context = {"a": 1}
    assert await _async_inject_context(context, async_func, 2) == 2


@pytest.mark.asyncio
async def test_sync_fn_to_async_fn():
    def sync_func():
        return "sync-to-async"

    def sync_gen():
        for i in range(3):
            yield f"sync-to-async-{i}"

    assert await _sync_fn_to_async_fn(sync_func) == "sync-to-async"
    async_gen = await _sync_fn_to_async_fn(sync_gen)
    assert isinstance(async_gen, AsyncGenerator)
