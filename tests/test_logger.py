import pytest
from unittest.mock import MagicMock, patch
from litserve.logger import Logger, _LoggerConnector


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
    with patch("litserve.logger._LoggerConnector._process_logger_queue", side_effect=KeyboardInterrupt), pytest.raises(
        KeyboardInterrupt
    ):
        logger_connector._process_logger_queue()
    test_logger.process("test_key", "test_value")
    assert test_logger.processed_data == ("test_key", "test_value")
