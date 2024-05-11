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
import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from litserve.examples.openai_spec_example import OpenAILitAPI, OpenAISpecWithHooks
from litserve.specs.openai import OpenAISpec
import litserve as ls

data = {
    "model": "",
    "messages": "string",
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "stream": False,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "user": "string",
}


@pytest.mark.asyncio()
async def test_openai_spec():
    spec = OpenAISpec()
    server = ls.LitServer(OpenAILitAPI(), spec=spec)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json=data, timeout=10)
        assert resp.status_code == 200, "Status code should be 200"

        assert resp.json()["choices"][0]["message"]["content"] == "This is a generated output", (
            "LitAPI predict response should match with " "the generated output"
        )


def test_openai_spec_with_hooks(capsys):
    output_text = "This is a generated output"
    server = ls.LitServer(OpenAILitAPI(), spec=OpenAISpec())
    server.app.lit_api.decode_request(data)
    response = server.app.lit_api.encode_response(output_text)
    assert response == {"text": output_text}, "encode_response should return a dict with text key and output value"
    capture = capsys.readouterr()
    assert "decode_request called from LitAPI" in capture.out, (
        "LitAPI.decode_request should be called when Spec " "doesn't implement decode_requst"
    )
    assert "encode_response called from LitAPI" in capture.out, (
        "LitAPI.encode_response should be called when " "Spec doesn't implement encode_response"
    )

    # Test with hooks, OpenAISpecWithHooks implements decode_request and encode_response
    server = ls.LitServer(OpenAILitAPI(), spec=OpenAISpecWithHooks())
    server.app.lit_api.decode_request(data)
    response = server.app.lit_api.encode_response(output_text)
    assert response == {"text": output_text}, "encode_response should return a dict with text key and output value"
    capture = capsys.readouterr()
    assert "decode_request called from Spec" in capture.out, (
        "LitSpec.decode_request should be called because " "OpenAISpecWithHooks implements decode_requst"
    )
    assert "encode_response called from Spec" in capture.out, (
        "LitSpec.encode_response should be called because" " OpenAISpecWithHooks implements encode_response"
    )
