import asyncio
import logging
import re
import time

import pytest
from asgi_lifespan import LifespanManager
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.callbacks.defaults import PredictionTimeLogger
from litserve.callbacks.defaults.metric_callback import RequestTracker
from litserve.utils import wrap_litserve_start

METRIC_CALLBACK_LOGGER = "litserve.callbacks.defaults.metric_callback"


async def run_simple_request(server, num_requests=1):
    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            responses = [ac.post("/predict", json={"input": 4.0}) for _ in range(num_requests)]
            responses = await asyncio.gather(*responses)
            for response in responses:
                assert response.json() == {"output": 16.0}, "Unexpected response"


class SlowAPI(ls.test_examples.SimpleLitAPI):
    def predict(self, x):
        time.sleep(1)
        return super().predict(x)


def test_callback_runner():
    cb_runner = CallbackRunner()
    assert cb_runner._callbacks == [], "Callbacks list must be empty"

    cb = PredictionTimeLogger()
    cb_runner._add_callbacks(cb)
    assert cb_runner._callbacks == [cb], "Callback not added to runner"


def test_callback(capfd):
    lit_api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(lit_api, callbacks=[PredictionTimeLogger()])

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}

    captured = capfd.readouterr()
    pattern = r"Prediction took \d+\.\d{2} seconds"
    assert re.search(pattern, captured.out), f"Expected pattern not found in output: {captured.out}"


def test_metric_logger(caplog):
    cb = PredictionTimeLogger()
    cb_runner = CallbackRunner()
    cb_runner._add_callbacks(cb)
    assert cb_runner._callbacks == [cb], "Callback not added to runner"

    with caplog.at_level(logging.INFO, logger=METRIC_CALLBACK_LOGGER):
        cb_runner.trigger_event(EventTypes.BEFORE_PREDICT.value, lit_api=None)
        cb_runner.trigger_event(EventTypes.AFTER_PREDICT.value, lit_api=None)

    pattern = r"Prediction took \d+\.\d{2} seconds"
    assert re.search(pattern, caplog.text), f"Expected pattern not found in logs: {caplog.text}"


@pytest.mark.asyncio
async def test_request_tracker(caplog):
    lit_api = SlowAPI()

    with caplog.at_level(logging.INFO, logger=METRIC_CALLBACK_LOGGER):
        server = ls.LitServer(lit_api, track_requests=False, callbacks=[RequestTracker()])
        await run_simple_request(server, 1)
    assert "Active requests: None" in caplog.text, f"Expected pattern not found in logs: {caplog.text}"

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=METRIC_CALLBACK_LOGGER):
        server = ls.LitServer(lit_api, track_requests=True, callbacks=[RequestTracker()])
        await run_simple_request(server, 4)
    assert "Active requests: 4" in caplog.text, f"Expected pattern not found in logs: {caplog.text}"


@pytest.mark.asyncio
async def test_request_tracker_with_spec(caplog):
    from litserve.specs.openai_embedding import OpenAIEmbeddingSpec
    from litserve.test_examples.openai_embedding_spec_example import TestEmbedAPI

    lit_api = TestEmbedAPI(spec=OpenAIEmbeddingSpec())
    server = ls.LitServer(lit_api, track_requests=True, callbacks=[RequestTracker()])

    with caplog.at_level(logging.INFO, logger=METRIC_CALLBACK_LOGGER), wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post("/v1/embeddings", json={"input": "test", "model": "test"})
            assert resp.status_code == 200

    assert "Active requests: 1" in caplog.text, f"Expected pattern not found in logs: {caplog.text}"


@pytest.mark.asyncio
async def test_request_tracker_with_openai_spec(caplog):
    from litserve.specs.openai import OpenAISpec
    from litserve.test_examples.openai_spec_example import TestAPI

    lit_api = TestAPI(spec=OpenAISpec())
    server = ls.LitServer(lit_api, track_requests=True, callbacks=[RequestTracker()])

    with caplog.at_level(logging.INFO, logger=METRIC_CALLBACK_LOGGER), wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "test"}], "model": "test"}
            )
            assert resp.status_code == 200

    assert "Active requests: 1" in caplog.text, f"Expected pattern not found in logs: {caplog.text}"
