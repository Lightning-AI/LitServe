import asyncio
import re
import time

import pytest
from asgi_lifespan import LifespanManager
from fastapi.testclient import TestClient
from httpx import AsyncClient

import litserve as ls
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.callbacks.defaults import PredictionTimeLogger
from litserve.callbacks.defaults.metric_callback import RequestTracker
from litserve.utils import wrap_litserve_start


async def run_simple_request(server, num_requests=1):
    with wrap_litserve_start(server) as server:
        async with LifespanManager(server.app) as manager, AsyncClient(app=manager.app, base_url="http://test") as ac:
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


def test_metric_logger(capfd):
    cb = PredictionTimeLogger()
    cb_runner = CallbackRunner()
    cb_runner._add_callbacks(cb)
    assert cb_runner._callbacks == [cb], "Callback not added to runner"
    cb_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=None)
    cb_runner.trigger_event(EventTypes.AFTER_PREDICT, lit_api=None)

    captured = capfd.readouterr()
    pattern = r"Prediction took \d+\.\d{2} seconds"
    assert re.search(pattern, captured.out), f"Expected pattern not found in output: {captured.out}"


@pytest.mark.asyncio
async def test_request_tracker(capfd):
    lit_api = SlowAPI()
    server = ls.LitServer(lit_api, track_requests=True, callbacks=[RequestTracker()])
    await run_simple_request(server, 4)

    captured = capfd.readouterr()
    assert "Active requests: 4" in captured.out, f"Expected pattern not found in output: {captured.out}"

    server = ls.LitServer(lit_api, track_requests=False, callbacks=[RequestTracker()])
    await run_simple_request(server, 1)

    captured = capfd.readouterr()
    assert "Active requests: None" in captured.out, f"Expected pattern not found in output: {captured.out}"
