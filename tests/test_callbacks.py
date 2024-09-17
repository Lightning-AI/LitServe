import re
from unittest.mock import MagicMock

import litserve as ls
from fastapi.testclient import TestClient

from litserve.callbacks import CallbackRunner, EventTypes
from litserve.callbacks.defaults import MetricLogger
from litserve.utils import wrap_litserve_start


def test_callback(capfd):
    lit_api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(lit_api, callbacks=[MetricLogger()])

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}

    captured = capfd.readouterr()
    pattern = r"Prediction took \d+\.\d{2} seconds"
    assert re.search(pattern, captured.out), f"Expected pattern not found in output: {captured.out}"


def test_metric_logger():
    cb = MetricLogger()
    cb_runner = CallbackRunner()
    cb_runner.add_callbacks(cb)
    assert cb_runner._callbacks == [cb], "Callback not added to runner"
    cb_runner.trigger_event(EventTypes.LITAPI_PREDICT_START)
    cb_runner.trigger_event(EventTypes.LITAPI_PREDICT_END)
    cb.on_litapi_predict_start = MagicMock()
    cb.on_litapi_predict_end = MagicMock()
    cb_runner.trigger_event(EventTypes.LITAPI_PREDICT_START)
    cb_runner.trigger_event(EventTypes.LITAPI_PREDICT_END)
    cb.on_litapi_predict_start.assert_called_once(), "on_litapi_predict_start not called"
    cb.on_litapi_predict_end.assert_called_once(), "on_litapi_predict_end not called"
