import pytest
import torch.nn
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from litserve.test_examples.openai_spec_example import (
    OpenAIWithUsage,
    OpenAIWithUsageEncodeResponse,
    OpenAIBatchingWithUsage,
)
from litserve.test_examples.simple_example import SimpleStreamAPI
from litserve.utils import wrap_litserve_start
import litserve as ls


@pytest.mark.asyncio()
async def test_simple_pytorch_api():
    api = ls.examples.SimpleTorchAPI()
    server = ls.LitServer(api, accelerator="cpu")
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            response = await ac.post("/predict", json={"input": 4.0})
            assert response.json() == {"output": 9.0}


@pytest.mark.asyncio()
async def test_simple_batched_api():
    api = ls.examples.SimpleBatchedAPI()
    server = ls.LitServer(api, max_batch_size=4, batch_timeout=0.1)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            response = await ac.post("/predict", json={"input": 4.0})
            assert response.json() == {"output": 16.0}


@pytest.mark.asyncio()
async def test_simple_api():
    api = ls.examples.SimpleLitAPI()
    server = ls.LitServer(api)
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
            response = await ac.post("/predict", json={"input": 4.0})
            assert response.json() == {"output": 16.0}


@pytest.mark.asyncio()
async def test_simple_api_without_server():
    api = ls.examples.SimpleLitAPI()
    api.setup(None)
    assert api.model is not None, "Model should be loaded after setup"
    assert api.predict(4) == 16, "Model should be able to predict"


@pytest.mark.asyncio()
async def test_simple_pytorch_api_without_server():
    api = ls.examples.SimpleTorchAPI()
    api.setup("cpu")
    assert api.model is not None, "Model should be loaded after setup"
    assert isinstance(api.model, torch.nn.Module)
    assert api.decode_request({"input": 4}) == 4, "Request should be decoded"
    assert api.predict(torch.Tensor([4])).cpu() == 9, "Model should be able to predict"
    assert api.encode_response(9) == {"output": 9}, "Response should be encoded"


@pytest.mark.asyncio()
async def test_simple_stream_api_without_server():
    api = SimpleStreamAPI()
    api.setup(None)
    assert api.model is not None, "Model should be loaded after setup"
    assert api.decode_request({"input": 4}) == 4, "Request should be decoded"
    assert list(api.predict(4)) == ["0: 4", "1: 4", "2: 4"], "Model should be able to predict"
    assert list(api.encode_response(["0: 4", "1: 4", "2: 4"])) == [
        {"output": "0: 4"},
        {"output": "1: 4"},
        {"output": "2: 4"},
    ], "Response should be encoded"


@pytest.mark.asyncio()
async def test_openai_with_usage():
    api = OpenAIWithUsage()
    api.setup(None)
    response = list(api.predict("10 + 6"))
    assert response == [
        {
            "role": "assistant",
            "content": "10 + 6 is equal to 16.",
            "prompt_tokens": 25,
            "completion_tokens": 10,
            "total_tokens": 35,
        }
    ], "Response should match expected output"


@pytest.mark.asyncio()
async def test_openai_with_usage_encode_response():
    api = OpenAIWithUsageEncodeResponse()
    api.setup(None)
    response = list(api.predict("10 + 6"))
    encoded_response = list(api.encode_response(response))
    assert encoded_response == [
        {"role": "assistant", "content": "10"},
        {"role": "assistant", "content": " +"},
        {"role": "assistant", "content": " "},
        {"role": "assistant", "content": "6"},
        {"role": "assistant", "content": " is"},
        {"role": "assistant", "content": " equal"},
        {"role": "assistant", "content": " to"},
        {"role": "assistant", "content": " "},
        {"role": "assistant", "content": "16"},
        {"role": "assistant", "content": "."},
        {"role": "assistant", "content": "", "prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
    ], "Encoded response should match expected output"


@pytest.mark.asyncio()
async def test_openai_batching_with_usage():
    api = OpenAIBatchingWithUsage()
    api.setup(None)
    inputs = ["10 + 6", "10 + 6"]
    batched_response = list(api.predict(inputs))
    assert batched_response == [["10 + 6 is equal to 16."] * 2], "Batched response should match expected output"
    encoded_response = list(api.encode_response(batched_response, [{"temperature": 1.0}, {"temperature": 1.0}]))
    assert encoded_response == [
        [
            {"role": "assistant", "content": "10 + 6 is equal to 16."},
            {"role": "assistant", "content": "10 + 6 is equal to 16."},
        ],
        [
            {"role": "assistant", "content": "", "prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
            {"role": "assistant", "content": "", "prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
        ],
    ], "Encoded batched response should match expected output"
