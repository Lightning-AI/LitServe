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
import time
from queue import Queue
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from asgi_lifespan import LifespanManager
from fastapi import Request, Response
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve import LitAPI, LitServer
from litserve.callbacks import CallbackRunner
from litserve.loops.base import _SENTINEL_VALUE, _StopLoopError, collate_requests
from litserve.loops.simple_loops import BatchedLoop
from litserve.transport.base import MessageTransport
from litserve.utils import LoopResponseType, wrap_litserve_start

NOOP_CB_RUNNER = CallbackRunner()


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.fill_(2.0)
        self.linear.bias.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class SimpleBatchLitAPI(LitAPI):
    def setup(self, device):
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request):
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def batch(self, inputs):
        assert len(inputs) == 2
        return torch.stack(inputs)

    def predict(self, x):
        assert len(x) == 2, "Expected two concurrent inputs to be batched"
        return self.model(x)

    def unbatch(self, output):
        assert output.shape[0] == 2
        return list(output)

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


class SimpleTorchAPI(LitAPI):
    def setup(self, device):
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request):
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def predict(self, x):
        assert x.shape == (1,), f"{x}"
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": float(output)}


@pytest.mark.asyncio
async def test_batched():
    api = SimpleBatchLitAPI(max_batch_size=2, batch_timeout=4)
    server = LitServer(api, accelerator="cpu", devices=1, timeout=10)

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            response1 = ac.post("/predict", json={"input": 4.0})
            response2 = ac.post("/predict", json={"input": 5.0})
            response1, response2 = await asyncio.gather(response1, response2)

    assert response1.json() == {"output": 9.0}
    assert response2.json() == {"output": 11.0}


@pytest.mark.asyncio
async def test_unbatched():
    api = SimpleTorchAPI(max_batch_size=1)
    api.request_timeout = 30
    api.pre_setup(spec=None)
    server = LitServer(api, accelerator="cpu", devices=1, timeout=10)
    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            response1 = ac.post("/predict", json={"input": 4.0})
            response2 = ac.post("/predict", json={"input": 5.0})
            response1, response2 = await asyncio.gather(response1, response2)

    assert response1.json() == {"output": 9.0}
    assert response2.json() == {"output": 11.0}


def test_request_timeout_connector():
    api = SimpleBatchLitAPI()
    LitServer(api, accelerator="cpu", devices=1, timeout=43.432)
    assert api.request_timeout == 43.432


def test_max_batch_size():
    with pytest.raises(ValueError, match="must be greater than 0"):
        SimpleBatchLitAPI(max_batch_size=0)

    with pytest.raises(ValueError, match="must be greater than 0"):
        SimpleBatchLitAPI(max_batch_size=-1)

    api = SimpleBatchLitAPI(max_batch_size=2, batch_timeout=5)
    with pytest.raises(ValueError, match="batch_timeout must be less than request_timeout"):
        LitServer(api, timeout=2)


def test_max_batch_size_warning():
    warning = "both batch and unbatch methods implemented, but the max_batch_size parameter was not set."
    with pytest.warns(
        UserWarning,
        match=warning,
    ):
        SimpleBatchLitAPI()

    # Test no warnings are raised when max_batch_size is set and max_batch_size is not set
    with (
        pytest.raises(pytest.fail.Exception),
        pytest.warns(
            UserWarning,
            match=warning,
        ),
    ):
        SimpleBatchLitAPI(max_batch_size=2)

    # Test no warning is set when LitAPI doesn't implement batch and unbatch
    with (
        pytest.raises(pytest.fail.Exception),
        pytest.warns(
            UserWarning,
            match=warning,
        ),
    ):
        SimpleTorchAPI()


def test_batch_predict_string_warning():
    api = ls.test_examples.SimpleBatchedAPI(max_batch_size=2, batch_timeout=0.1)
    api.request_timeout = 30
    api.pre_setup(spec=None)
    api.predict = MagicMock(return_value="This is a string")

    mock_input = torch.tensor([[1.0], [2.0]])

    # Simulate the behavior in run_batched_loop
    y = api.predict(mock_input)
    with pytest.warns(
        UserWarning,
        match="When batching is enabled, 'predict' must return a list to handle multiple inputs correctly.",
    ):
        api.unbatch(y)


class FakeResponseQueue:
    def put(self, *args, block=True, timeout=None):
        raise StopIteration("exit loop")


class FakeTransport(MessageTransport):
    def __init__(self):
        self.responses = []

    async def areceive(self, **kwargs) -> dict:
        raise NotImplementedError("This is a fake transport")

    def send(self, response, consumer_id: int):
        self.responses.append(response)


def test_batched_loop():
    requests_queue = Queue()
    response_queue_id = 0
    requests_queue.put((response_queue_id, "uuid-1234", time.monotonic(), {"input": 4.0}))
    requests_queue.put((response_queue_id, "uuid-1235", time.monotonic(), {"input": 5.0}))
    requests_queue.put(_SENTINEL_VALUE)

    lit_api_mock = MagicMock()
    lit_api_mock.request_timeout = 2
    lit_api_mock.max_batch_size = 2
    lit_api_mock.batch_timeout = 4
    lit_api_mock.decode_request = MagicMock(side_effect=lambda x: x["input"])
    lit_api_mock.batch = MagicMock(side_effect=lambda x: x)
    lit_api_mock.predict = MagicMock(side_effect=lambda x: [16.0, 25.0])
    lit_api_mock.unbatch = MagicMock(side_effect=lambda x: x)
    lit_api_mock.encode_response = MagicMock(side_effect=lambda x: {"output": x})

    loop = BatchedLoop()
    transport = FakeTransport()
    loop.run_batched_loop(
        lit_api_mock,
        requests_queue,
        transport=transport,
        callback_runner=NOOP_CB_RUNNER,
    )

    assert len(transport.responses) == 2, "response queue should have 2 responses"
    assert transport.responses[0] == ("uuid-1234", ({"output": 16.0}, "OK", LoopResponseType.REGULAR))
    assert transport.responses[1] == ("uuid-1235", ({"output": 25.0}, "OK", LoopResponseType.REGULAR))

    lit_api_mock.batch.assert_called_once()
    lit_api_mock.batch.assert_called_once_with([4.0, 5.0])
    lit_api_mock.unbatch.assert_called_once()


@pytest.mark.parametrize(
    ("batch_timeout", "batch_size"),
    [
        pytest.param(0, 2),
        pytest.param(0, 1000),
        pytest.param(0.1, 2),
        pytest.param(1000, 2),
        pytest.param(0.1, 1000),
    ],
)
def test_collate_requests(batch_timeout, batch_size):
    api = ls.test_examples.SimpleBatchedAPI(max_batch_size=batch_size, batch_timeout=batch_timeout)
    api.request_timeout = 5
    request_queue = Queue()
    for i in range(batch_size):
        request_queue.put((i, f"uuid-abc-{i}", time.monotonic(), i))  # response_queue_id, uid, timestamp, x_enc
    payloads, timed_out_uids = collate_requests(api, request_queue)
    assert len(payloads) == batch_size, f"Should have {batch_size} payloads, got {len(payloads)}"
    assert len(timed_out_uids) == 0, "No timed out uids"


def test_collate_requests_sentinel():
    api = ls.test_examples.SimpleBatchedAPI(max_batch_size=2, batch_timeout=0)
    api.request_timeout = 5
    request_queue = Queue()
    request_queue.put(_SENTINEL_VALUE)
    with pytest.raises(_StopLoopError, match="Received sentinel value, stopping loop"):
        collate_requests(api, request_queue)


class BatchSizeMismatchAPI(SimpleBatchLitAPI):
    def predict(self, x):
        assert len(x) == 2, "Expected two concurrent inputs to be batched"
        return self.model(x)  # returns a list of length same as len(x)

    def unbatch(self, output):
        return [output]  # returns a list of length 1


@pytest.mark.asyncio
async def test_batch_size_mismatch():
    api = BatchSizeMismatchAPI(max_batch_size=2, batch_timeout=4)
    api.request_timeout = 30
    api.pre_setup(spec=None)
    server = LitServer(api, accelerator="cpu", devices=1, timeout=10)

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            response1 = ac.post("/predict", json={"input": 4.0})
            response2 = ac.post("/predict", json={"input": 5.0})
            response1, response2 = await asyncio.gather(response1, response2)
        assert response1.status_code == 500
        assert response2.status_code == 500
        assert response1.json() == {"detail": "Batch size mismatch"}, "unbatch a list of length 1 when batch size is 2"
        assert response2.json() == {"detail": "Batch size mismatch"}, "unbatch a list of length 1 when batch size is 2"
