import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from litserve.examples.openai_spec_example import OpenAILitAPI
from litserve.specs.openai import OpenAISpec
import litserve as ls


@pytest.mark.asyncio()
async def test_openai_spec():
    spec = OpenAISpec()
    server = ls.LitServer(OpenAILitAPI(), spec=spec)
    async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
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
        resp = await ac.post("/v1/chat/completions", json=data, timeout=10)
        assert resp.status_code == 200, "Status code should be 200"
        assert resp.json()["choices"][0]["message"]["content"] == "This is a generated output", (
            "LitAPI predict response should match with " "the generated output"
        )
