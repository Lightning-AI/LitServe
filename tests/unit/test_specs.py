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
import json

import pytest
from asgi_lifespan import LifespanManager
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.specs.openai import ChatCompletionRequest, ChatMessage, OpenAISpec
from litserve.test_examples.openai_spec_example import (
    OpenAIBatchingWithUsage,
    OpenAIWithUsage,
    OpenAIWithUsageEncodeResponse,
    TestAPI,
    TestAPIWithCustomEncode,
    TestAPIWithStructuredOutput,
    TestAPIWithToolCalls,
)
from litserve.utils import wrap_litserve_start


class TestOpenAISpecAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        assert isinstance(x, ChatCompletionRequest), "decode_request returns a ChatCompletionRequest"
        for e in ["This", "is", "a", "generated", "output"]:
            yield e + " "


class TestAsyncAPI(TestOpenAISpecAPI):
    async def predict(self, x):
        assert isinstance(x, ChatCompletionRequest), "decode_request returns a ChatCompletionRequest"
        for e in ["This", "is", "a", "generated", "output"]:
            yield e + " "


@pytest.mark.parametrize(
    "api", [TestOpenAISpecAPI(spec=OpenAISpec()), TestAsyncAPI(enable_async=True, spec=OpenAISpec())]
)
@pytest.mark.asyncio
async def test_openai_spec(openai_request_data, api):
    server = ls.LitServer(api)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app),
            base_url="http://test",
        ) as ac:
            openai_request_data["stream"] = True
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            messages = []
            async for chunk in resp.aiter_lines():
                if not chunk.startswith("data: "):
                    continue
                content = chunk[6:].strip()
                if content == "[DONE]" or not content:
                    break
                try:
                    chunk = json.loads(content)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content_piece = delta.get("content")
                    if content_piece is not None:
                        messages.append(content_piece)
                except json.JSONDecodeError:
                    continue  # Optionally log or handle bad JSON chunks

            final_output = "".join(messages)
            assert final_output == "This is a generated output ", f"final_output: {final_output}"


# OpenAIWithUsage
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api", "batch_size"),
    [
        (OpenAIWithUsage(), 1),
        (OpenAIWithUsageEncodeResponse(), 1),
        (OpenAIBatchingWithUsage(), 2),
    ],
)
async def test_openai_token_usage(api, batch_size, openai_request_data, openai_response_data):
    server = ls.LitServer(api, spec=ls.OpenAISpec(), max_batch_size=batch_size, batch_timeout=0.01)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            assert content == "10 + 6 is equal to 16.", "LitAPI predict response should match with the generated output"
            assert result["usage"] == openai_response_data["usage"]

            # with streaming
            openai_request_data["stream"] = True
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            assert result["usage"] == openai_response_data["usage"]


class OpenAIWithUsagePerToken(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        for i in range(1, 6):
            yield {
                "role": "assistant",
                "content": f"{i}",
                "prompt_tokens": 0,
                "completion_tokens": 1,
                "total_tokens": 1,
            }


# OpenAIWithUsagePerToken
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api", "batch_size"),
    [
        (OpenAIWithUsagePerToken(), 1),
    ],
)
async def test_openai_per_token_usage(api, batch_size, openai_request_data, openai_response_data):
    server = ls.LitServer(api, spec=ls.OpenAISpec(), max_batch_size=batch_size, batch_timeout=0.01)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            assert content == "12345", "LitAPI predict response should match with the generated output"
            assert result["usage"]["completion_tokens"] == 5, "API yields 5 tokens"

            # with streaming
            openai_request_data["stream"] = True
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            assert result["usage"]["completion_tokens"] == 5, "API yields 5 tokens"


@pytest.mark.asyncio
async def test_openai_spec_with_image(openai_request_data_with_image):
    server = ls.LitServer(TestAPI(), spec=OpenAISpec())
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data_with_image, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"

            assert resp.json()["choices"][0]["message"]["content"] == "This is a generated output", (
                "LitAPI predict response should match with the generated output"
            )


@pytest.mark.asyncio
async def test_openai_spec_with_audio(openai_request_data_with_audio_wav, openai_request_data_with_audio_flac):
    server = ls.LitServer(TestAPI(), spec=OpenAISpec())

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data_with_audio_wav, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"

            assert resp.json()["choices"][0]["message"]["content"] == "This is a generated output", (
                "LitAPI predict response should match with the generated output"
            )

            # test for unsupported audio format
            resp = await ac.post("/v1/chat/completions", json=openai_request_data_with_audio_flac, timeout=10)
            assert resp.status_code == 422, "Status code should be 422"
            errors = resp.json()["detail"]
            assert any(error["msg"] == "Input should be 'wav' or 'mp3'" for error in errors), (
                "Error message for unsupported audio format should be present"
            )


@pytest.mark.asyncio
async def test_override_encode(openai_request_data):
    server = ls.LitServer(TestAPIWithCustomEncode(), spec=OpenAISpec())
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"

            assert resp.json()["choices"][0]["message"]["content"] == "This is a custom encoded output", (
                "LitAPI predict response should match with the generated output"
            )


@pytest.mark.asyncio
async def test_openai_spec_with_tools(openai_request_data_with_tools):
    spec = OpenAISpec()
    server = ls.LitServer(TestAPIWithToolCalls(), spec=spec)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data_with_tools, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            assert resp.json()["choices"][0]["message"]["content"] == "", (
                "LitAPI predict response should match with the generated output"
            )
            assert resp.json()["choices"][0]["message"]["tool_calls"] == [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "function_1", "arguments": '{"arg_1": "arg_1_value"}'},
                    "index": 0,
                }
            ], "LitAPI predict response should match with the generated output"


@pytest.mark.asyncio
async def test_openai_spec_with_response_format(openai_request_data_with_response_format):
    spec = OpenAISpec()
    server = ls.LitServer(TestAPIWithStructuredOutput(), spec=spec)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data_with_response_format, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"
            assert (
                resp.json()["choices"][0]["message"]["content"]
                == '{"name": "Science Fair", "date": "Friday", "participants": ["Alice", "Bob"]}'
            ), "LitAPI predict response should match with the generated output"


class MetadataRequiredAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device

    def decode_request(self, request):
        return request

    def predict(self, request):
        metadata = request.metadata
        if not metadata or "user_id" not in metadata:
            raise HTTPException(status_code=500, detail="Missing required metadata")
        yield "ok"


@pytest.mark.asyncio
async def test_openai_spec_metadata(openai_request_data_with_metadata):
    server = ls.LitServer(MetadataRequiredAPI(), spec=OpenAISpec())

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data_with_metadata)
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_openai_spec_metadata_required_fail(openai_request_data):
    server = ls.LitServer(MetadataRequiredAPI(), spec=OpenAISpec())

    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data)
            assert resp.status_code == 500
            assert "Missing required metadata" in resp.text


class TestAPIWithReasoningEffort(TestAPI):
    def encode_response(self, output, context):
        yield ChatMessage(
            role="assistant",
            content=f"This is a generated output with reasoning effort: {context.get('reasoning_effort', None)}",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["low", "medium", "high", None, "random"])
async def test_openai_spec_reasoning_effort(reasoning_effort, openai_request_data):
    server = ls.LitServer(TestAPIWithReasoningEffort(), spec=OpenAISpec())
    openai_request_data["reasoning_effort"] = reasoning_effort
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data)
            if reasoning_effort == "random":
                assert resp.status_code == 422  # as random is not a valid reasoning effort
            else:
                assert resp.status_code == 200
                assert (
                    resp.json()["choices"][0]["message"]["content"]
                    == f"This is a generated output with reasoning effort: {reasoning_effort}"
                )


class IncorrectAPI1(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        return "This is a generated output"


class IncorrectAPI2(IncorrectAPI1):
    def predict(self, x):
        yield "This is a generated output"

    def encode_response(self, output):
        return ChatMessage(role="assistant", content="This is a generated output")


@pytest.mark.asyncio
async def test_openai_spec_validation(openai_request_data):
    with pytest.raises(ValueError, match="predict is not a generator"):
        ls.LitServer(IncorrectAPI1(), spec=OpenAISpec())

    with pytest.raises(ValueError, match="encode_response is not a generator"):
        ls.LitServer(IncorrectAPI2(), spec=OpenAISpec())


class PrePopulatedAPI(ls.LitAPI):
    def setup(self, device):
        self.sentence = ["This", " is", " a", " sample", " response"]

    def predict(self, prompt, context):
        for count, token in enumerate(self.sentence, start=1):
            yield token
            if count >= context["max_completion_tokens"]:
                return


@pytest.mark.asyncio
async def test_oai_prepopulated_context(openai_request_data):
    openai_request_data["max_completion_tokens"] = 3
    spec = OpenAISpec()
    server = ls.LitServer(PrePopulatedAPI(), spec=spec)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.json()["choices"][0]["message"]["content"] == "This is a", (
                "OpenAISpec must return only 3 tokens as specified using `max_completion_tokens` parameter"
            )


class WrongLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, prompt):
        yield "This is a sample generated text"
        raise HTTPException(501, "test LitAPI.predict error")


@pytest.mark.asyncio
async def test_fail_http(openai_request_data):
    server = ls.LitServer(WrongLitAPI(spec=ls.OpenAISpec()))
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            res = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert res.status_code == 501, f"Server raises 501 error: {res.content}"
            assert res.text == '{"detail":"test LitAPI.predict error"}'


class IncorrectAsyncAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    async def decode_request(self, request):
        return request

    async def predict(self, x):
        return "This is a generated output"

    async def encode_response(self, output):
        return ChatMessage(role="assistant", content="This is a generated output")


class IncorrectDecodeAsyncAPI(IncorrectAsyncAPI):
    def decode_request(self, request):
        return request


class IncorrectEncodeAsyncAPI(IncorrectAsyncAPI):
    async def predict(self, x):
        yield "This is a generated output"


@pytest.mark.asyncio
async def test_openai_spec_asyncapi_predict_validation():
    with pytest.raises(ValueError, match="predict must be an async generator"):
        ls.LitServer(IncorrectAsyncAPI(enable_async=True), spec=OpenAISpec())


@pytest.mark.asyncio
def test_openai_spec_asyncapi_encode_response_validation():
    with pytest.raises(ValueError, match="encode_response is neither a generator nor an async generator"):
        ls.LitServer(IncorrectEncodeAsyncAPI(enable_async=True), spec=OpenAISpec())


@pytest.mark.asyncio
def test_openai_asyncapi_enable_async_flag_validation():
    with pytest.raises(ValueError, match="'enable_async' is not set in LitAPI."):
        ls.LitServer(IncorrectAsyncAPI(enable_async=False), spec=OpenAISpec())


class DecodeNotImplementedAsyncOpenAILitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    async def predict(self, x):
        yield "This is a generated output"

    async def encode_response(self, output):
        yield {"role": "assistant", "content": output}


class AsyncOpenAILitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None
        self.sentence = ["This", " is", " a", " sample", " response"]

    async def decode_request(self, request):
        return request

    async def predict(self, x):
        for token in self.sentence:
            yield token

    async def encode_response(self, output_stream, context):
        async for output in output_stream:
            yield {"role": "assistant", "content": output}


@pytest.mark.asyncio
async def test_openai_spec_with_async_litapi(openai_request_data):
    server = ls.LitServer(AsyncOpenAILitAPI(enable_async=True), spec=OpenAISpec())
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=openai_request_data, timeout=10)
            assert resp.status_code == 200, "Status code should be 200"

            assert resp.json()["choices"][0]["message"]["content"] == "This is a sample response", (
                "LitAPI predict response should match with the generated output"
            )
