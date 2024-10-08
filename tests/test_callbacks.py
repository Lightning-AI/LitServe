import re

from fastapi.testclient import TestClient

import litserve as ls
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.callbacks.defaults import PredictionTimeLogger
from litserve.utils import wrap_litserve_start


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
