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

import pytest
from asgi_lifespan import LifespanManager
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.utils import wrap_litserve_start


class MinimalAsyncAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    async def predict(self, x):
        y = x["input"] ** 2
        return {"output": y}


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_async", [None, True])
async def test_async_api(enable_async):
    server = ls.LitServer(MinimalAsyncAPI(enable_async=enable_async))
    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            response = await ac.post("/predict", json={"input": 2})
            assert response.json() == {"output": 4}


class SyncPipeline(ls.LitAPI):
    def decode_request(self, request, context):
        context["input"] = request["input"]
        return request["input"]

    def predict(self, x, context):
        assert context["input"] == x
        context["predicted"] = True
        return x**2

    def encode_response(self, output, context):
        assert context["predicted"]
        return {"input": context["input"], "output": output}


class AsyncDecode:
    async def decode_request(self, request, context):
        await asyncio.sleep(0)
        return super().decode_request(request, context)


class AsyncPredict:
    async def predict(self, x, context):
        await asyncio.sleep(0)
        return super().predict(x, context)


class AsyncEncode:
    async def encode_response(self, output, context):
        await asyncio.sleep(0)
        return super().encode_response(output, context)


class AsyncDecodePipeline(AsyncDecode, SyncPipeline):
    pass


class AsyncPredictPipeline(AsyncPredict, SyncPipeline):
    pass


class AsyncEncodePipeline(AsyncEncode, SyncPipeline):
    pass


class AsyncDecodePredictPipeline(AsyncDecode, AsyncPredict, SyncPipeline):
    pass


class AsyncDecodeEncodePipeline(AsyncDecode, AsyncEncode, SyncPipeline):
    pass


class AsyncPredictEncodePipeline(AsyncPredict, AsyncEncode, SyncPipeline):
    pass


class AsyncPipeline(AsyncDecode, AsyncPredict, AsyncEncode, SyncPipeline):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_async", [None, True])
@pytest.mark.parametrize(
    "api_cls",
    [
        SyncPipeline,
        AsyncDecodePipeline,
        AsyncPredictPipeline,
        AsyncEncodePipeline,
        AsyncDecodePredictPipeline,
        AsyncDecodeEncodePipeline,
        AsyncPredictEncodePipeline,
        AsyncPipeline,
    ],
)
async def test_mixed_async_pipeline(api_cls, enable_async):
    api = api_cls() if enable_async is None else api_cls(enable_async=enable_async)
    server = ls.LitServer(api, accelerator="cpu", fast_queue=False)
    with wrap_litserve_start(server):
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client,
        ):
            responses = await asyncio.gather(*(client.post("/predict", json={"input": x}) for x in range(4)))
            for x, response in enumerate(responses):
                assert response.status_code == 200
                assert response.json() == {"input": x, "output": x**2}


class MixedErrorPipeline(AsyncEncodePipeline):
    def predict(self, x, context):
        if x == 1:
            raise HTTPException(422, "sync prediction failed")
        if x == 2:
            raise ValueError("prediction failed")
        return super().predict(x, context)

    async def encode_response(self, output, context):
        if output == 9:
            raise HTTPException(409, "async encoding failed")
        return await super().encode_response(output, context)


@pytest.mark.asyncio
async def test_mixed_async_errors():
    server = ls.LitServer(MixedErrorPipeline(), accelerator="cpu", fast_queue=False)
    with wrap_litserve_start(server):
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client,
        ):
            for value, status, detail in [
                (1, 422, "sync prediction failed"),
                (2, 500, None),
                (3, 409, "async encoding failed"),
                (4, 200, None),
            ]:
                response = await client.post("/predict", json={"input": value})
                assert response.status_code == status
                if detail:
                    assert response.json()["detail"] == detail
                if status == 200:
                    assert response.json() == {"input": value, "output": value**2}


class SyncPredictAsyncStream(ls.LitAPI):
    def predict(self, x):
        yield x["input"]
        yield x["input"] + 1

    async def encode_response(self, output):
        async for item in output:
            yield {"output": item}


class AsyncPredictStream(SyncPredictAsyncStream):
    async def predict(self, x):
        yield x["input"]
        yield x["input"] + 1


class AsyncDecodeStream(SyncPredictAsyncStream):
    async def decode_request(self, request):
        return request


@pytest.mark.asyncio
@pytest.mark.parametrize("api_cls", [SyncPredictAsyncStream, AsyncPredictStream, AsyncDecodeStream])
async def test_mixed_async_stream(api_cls):
    server = ls.LitServer(api_cls(stream=True), accelerator="cpu", fast_queue=False)
    with wrap_litserve_start(server):
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client,
        ):
            response = await client.post("/predict", json={"input": 2})
            assert response.status_code == 200
            assert [json.loads(line) for line in response.text.splitlines()] == [{"output": 2}, {"output": 3}]
