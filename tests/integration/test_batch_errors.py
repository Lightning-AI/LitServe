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

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from litserve import LitAPI, LitServer
from litserve.utils import PickleableHTTPException, wrap_litserve_start


class BatchErrorAPI(LitAPI):
    def __init__(self, error_stage):
        super().__init__(max_batch_size=3, batch_timeout=1)
        self.error_stage = error_stage

    def setup(self, device):
        pass

    def decode_request(self, request):
        value = request["input"]
        if self.error_stage == "decode" and value < 0:
            raise PickleableHTTPException(422, "invalid input")
        return value

    def batch(self, inputs):
        return inputs

    def predict(self, inputs):
        return [{"output": value, "batch_size": len(inputs)} for value in inputs]

    def unbatch(self, output):
        return output

    def encode_response(self, output, context):
        if self.error_stage == "encode" and output["output"] < 0:
            raise PickleableHTTPException(422, "invalid input")
        return {**output, "header": context["headers"]["x-request-id"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("error_stage", ["decode", "encode"])
async def test_batch_request_error_isolation(error_stage):
    server = LitServer(BatchErrorAPI(error_stage), accelerator="cpu", devices=1, timeout=10)
    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client,
        ):
            responses = await asyncio.gather(
                *[
                    client.post("/predict", json={"input": value}, headers={"x-request-id": str(value)})
                    for value in [3, -1, 5]
                ]
            )

    assert [response.status_code for response in responses] == [200, 422, 200]
    assert responses[1].json() == {"detail": "invalid input"}
    batch_size = 2 if error_stage == "decode" else 3
    for response, value in [(responses[0], 3), (responses[2], 5)]:
        assert response.json() == {"output": value, "batch_size": batch_size, "header": str(value)}
