import re

import litserve as ls
from fastapi.testclient import TestClient

from litserve.callbacks import CallbackRunner, EventTypes
from litserve.callbacks.defaults import PredictionTimeLogger
from litserve.utils import wrap_litserve_start


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
    cb_runner.add_callbacks(cb)
    assert cb_runner._callbacks == [cb], "Callback not added to runner"
    cb_runner.trigger_event(EventTypes.LITAPI_PREDICT_START, lit_api=None)
    cb_runner.trigger_event(EventTypes.LITAPI_PREDICT_END, lit_api=None)

    captured = capfd.readouterr()
    pattern = r"Prediction took \d+\.\d{2} seconds"
    assert re.search(pattern, captured.out), f"Expected pattern not found in output: {captured.out}"
