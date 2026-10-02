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
import json
import os

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from litserve import LitAPI, LitServer
from litserve.utils import wrap_litserve_start


class BinaryAPI(LitAPI):
    def decode_request(self, request: bytes, context):
        assert type(request) is bytes
        return {"body": request.hex(), "content_type": context["content_type"], "pid": os.getpid()}

    def predict(self, x, context):
        if self.max_batch_size > 1:
            assert [item["content_type"] for item in x] == [item["content_type"] for item in context]
            return [dict(item, batch_size=len(x)) for item in x]
        assert x["content_type"] == context["content_type"]
        return dict(x, batch_size=1)


class AsyncBinaryAPI(BinaryAPI):
    async def decode_request(self, request: bytes, context):
        return super().decode_request(request, context)

    async def predict(self, x, context):
        return super().predict(x, context)

    async def encode_response(self, output):
        return output


class StreamingBinaryAPI(BinaryAPI):
    def predict(self, x, context):
        yield super().predict(x, context)

    def encode_response(self, output):
        yield from output


class AsyncStreamingBinaryAPI(BinaryAPI):
    async def decode_request(self, request: bytes, context):
        return super().decode_request(request, context)

    async def predict(self, x, context):
        yield super().predict(x, context)

    async def encode_response(self, output):
        async for item in output:
            yield item


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_cls", "stream", "enable_async", "max_batch_size"),
    [
        (BinaryAPI, False, False, 1),
        (AsyncBinaryAPI, False, True, 1),
        (StreamingBinaryAPI, True, False, 1),
        (AsyncStreamingBinaryAPI, True, True, 1),
        (BinaryAPI, False, False, 2),
        (StreamingBinaryAPI, True, False, 2),
    ],
    ids=["single", "async", "stream", "async-stream", "batch", "batch-stream"],
)
async def test_binary_request_worker_roundtrip(api_cls, stream, enable_async, max_batch_size):
    api = api_cls(
        stream=stream,
        enable_async=enable_async,
        max_batch_size=max_batch_size,
        batch_timeout=1 if max_batch_size > 1 else 0,
    )
    server = LitServer(api, accelerator="cpu", devices=1, timeout=20)
    requests = [
        (b"\x00\xff\x80\x01", "application/x-recordio-protobuf"),
        (bytes(range(256)), 'Application/Octet-Stream; version="1"'),
        (b"", "application/octet-stream"),
        (b"\xff\x00", ""),
        (b'{"input": 4}', "application/json"),
        (b"input=4", "application/x-www-form-urlencoded"),
    ]
    with wrap_litserve_start(server):
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client,
        ):
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/predict", content=body, headers={"Content-Type": content_type} if content_type else {}
                    )
                    for body, content_type in requests
                ]
            )

    for response, (body, content_type) in zip(responses, requests):
        assert response.status_code == 200, response.text
        result = json.loads(response.text)
        assert result["body"] == body.hex()
        assert result["content_type"] == content_type
        assert result["pid"] != os.getpid()
        assert result["batch_size"] == max_batch_size


class BinaryAPIWithoutContext(LitAPI):
    def decode_request(self, request: bytes):
        assert type(request) is bytes
        return request.hex()

    def predict(self, x):
        return {"body": x}


@pytest.mark.asyncio
async def test_binary_request_without_context():
    server = LitServer(BinaryAPIWithoutContext(), accelerator="cpu", devices=1)
    with wrap_litserve_start(server):
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client,
        ):
            response = await client.post("/predict", content=b"\xff\x00")
    assert response.status_code == 200
    assert response.json() == {"body": "ff00"}
