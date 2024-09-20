import contextlib
import os

import pytest
from fastapi.testclient import TestClient

from unittest.mock import MagicMock, patch
from litserve.loggers import Logger, _LoggerConnector

import litserve as ls
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


def test_logger_set_connector(test_logger, logger_connector):
    assert test_logger._connector == logger_connector


def test_logger_mount(test_logger):
    mock_app = MagicMock()
    test_logger.mount("/test", mock_app)
    assert test_logger._config["mount"]["path"] == "/test"
    assert test_logger._config["mount"]["app"] == mock_app


def test_connector_add_logger(logger_connector):
    new_logger = TestLogger()
    logger_connector.add_logger(new_logger)
    assert new_logger in logger_connector._loggers
    assert new_logger._connector == logger_connector


def test_connector_mount(mock_lit_server, test_logger, logger_connector):
    mock_app = MagicMock()
    test_logger.mount("/test", mock_app)
    logger_connector.add_logger(test_logger)
    mock_lit_server.app.mount.assert_called_with("/test", mock_app)


def test_process_logger_queue(mock_lit_server, logger_connector, test_logger):
    with patch("litserve.loggers._LoggerConnector._process_logger_queue", side_effect=KeyboardInterrupt), pytest.raises(
        KeyboardInterrupt
    ):
        logger_connector._process_logger_queue()
    test_logger.process("test_key", "test_value")
    assert test_logger.processed_data == ("test_key", "test_value")


class LoggerAPI(ls.test_examples.SimpleLitAPI):
    def predict(self, input):
        result = super().predict(input)
        for i in range(1, 5):
            self.log("time", i * 0.1)
        return result


def test_logger_queue():
    api = LoggerAPI()
    server = ls.LitServer(api)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
        metric = server.logger_queue.get(timeout=1)
        assert metric == ("time", 0.1), "Expected metric not found in logger queue"


class FileLogger(ls.Logger):
    def process(self, key, value):
        with open("test_logger_temp.txt", "a+") as f:
            f.write(f"{key}: {value:.1f}\n")


def test_logger_with_api():
    # Connect with a Logger
    api = LoggerAPI()
    server = ls.LitServer(api, loggers=[FileLogger()])
    with contextlib.suppress(Exception):
        os.remove("test_logger_temp.txt")
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
        with open("test_logger_temp.txt") as f:
            data = f.readlines()
            assert data == [
                "time: 0.1\n",
                "time: 0.2\n",
                "time: 0.3\n",
                "time: 0.4\n",
            ], f"Expected metric not found in logger file {data}"
    os.remove("test_logger_temp.txt")
