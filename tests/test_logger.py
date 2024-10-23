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
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import litserve as ls
from litserve.loggers import Logger, _LoggerConnector
from litserve.utils import wrap_litserve_start


class TestLogger(Logger):
    def process(self, key, value):
        self.processed_data = (key, value)


@pytest.fixture
def mock_lit_server():
    mock_server = MagicMock()
    mock_server.log_queue.get = MagicMock(return_value=("test_key", "test_value"))
    return mock_server


@pytest.fixture
def test_logger():
    return TestLogger()


@pytest.fixture
def logger_connector(mock_lit_server, test_logger):
    return _LoggerConnector(mock_lit_server, [test_logger])


def test_logger_mount(test_logger):
    mock_app = MagicMock()
    test_logger.mount("/test", mock_app)
    assert test_logger._config["mount"]["path"] == "/test"
    assert test_logger._config["mount"]["app"] == mock_app


def test_connector_add_logger(logger_connector):
    new_logger = TestLogger()
    logger_connector.add_logger(new_logger)
    assert new_logger in logger_connector._loggers


def test_connector_mount(mock_lit_server, test_logger, logger_connector):
    mock_app = MagicMock()
    test_logger.mount("/test", mock_app)
    logger_connector.add_logger(test_logger)
    mock_lit_server.app.mount.assert_called_with("/test", mock_app)


def test_invalid_loggers():
    _LoggerConnector(None, TestLogger())
    with pytest.raises(ValueError, match="Logger must be an instance of litserve.Logger"):
        _ = _LoggerConnector(None, [MagicMock()])

    with pytest.raises(ValueError, match="loggers must be a list or an instance of litserve.Logger"):
        _ = _LoggerConnector(None, MagicMock())


class LoggerAPI(ls.test_examples.SimpleLitAPI):
    def predict(self, input):
        result = super().predict(input)
        for i in range(1, 5):
            self.log("time", i * 0.1)
        return result


def test_server_wo_logger():
    api = LoggerAPI()
    server = ls.LitServer(api)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}


class FileLogger(ls.Logger):
    def __init__(self, path="test_logger_temp.txt"):
        super().__init__()
        self.path = path

    def process(self, key, value):
        with open(self.path, "a+") as f:
            f.write(f"{key}: {value:.1f}\n")


def test_logger_with_api(tmpdir):
    path = str(tmpdir / "test_logger_temp.txt")
    api = LoggerAPI()
    server = ls.LitServer(api, loggers=[FileLogger(path)])
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
        # Wait for FileLogger to write to file
        time.sleep(0.5)
        with open(path) as f:
            data = f.readlines()
            assert data == [
                "time: 0.1\n",
                "time: 0.2\n",
                "time: 0.3\n",
                "time: 0.4\n",
            ], f"Expected metric not found in logger file {data}"


class PredictionTimeLogger(ls.Callback):
    def on_after_predict(self, lit_api):
        for i in range(1, 5):
            lit_api.log("time", i * 0.1)


def test_logger_with_callback(tmp_path):
    path = str(tmp_path / "test_logger_temp.txt")
    api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(api, loggers=[FileLogger(path)], callbacks=[PredictionTimeLogger()])
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
        # Wait for FileLogger to write to file
        time.sleep(0.5)
        with open(path) as f:
            data = f.readlines()
            assert data == [
                "time: 0.1\n",
                "time: 0.2\n",
                "time: 0.3\n",
                "time: 0.4\n",
            ], f"Expected metric not found in logger file {data}"


class NonPickleableLogger(ls.Logger):
    # This is a logger that contains a non-picklable resource
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()  # Non-picklable resource

    def process(self, key, value):
        with self._lock:
            print(f"Logged {key}: {value}", flush=True)


class PickleTestAPI(ls.test_examples.SimpleLitAPI):
    def predict(self, x):
        self.log("my-key", x)
        return super().predict(x)


def test_pickle_safety(capfd):
    api = PickleTestAPI()
    server = ls.LitServer(api, loggers=NonPickleableLogger())
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
        time.sleep(0.5)
        captured = capfd.readouterr()
        assert "Logged my-key: 4.0" in captured.out, f"Expected log not found in captured output {captured}"
