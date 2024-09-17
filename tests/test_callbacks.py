import re

import litserve as ls
from fastapi.testclient import TestClient

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
