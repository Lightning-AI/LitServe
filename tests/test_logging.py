import io
import logging

import pytest

from litserve.utils import add_log_handler, configure_logging, set_log_level


@pytest.fixture
def log_stream():
    return io.StringIO()


def test_configure_logging(log_stream):
    # Configure logging with test stream
    configure_logging(level=logging.DEBUG, stream=log_stream)

    # Get logger and log a test message
    logger = logging.getLogger("litserve")
    test_message = "Test debug message"
    logger.debug(test_message)

    # Verify log output
    log_contents = log_stream.getvalue()
    assert test_message in log_contents
    assert "DEBUG" in log_contents
    assert logger.propagate is False


def test_set_log_level():
    # Set log level to WARNING
    set_log_level(logging.WARNING)

    # Verify logger level
    logger = logging.getLogger("litserve")
    assert logger.level == logging.WARNING


def test_add_log_handler():
    # Create and add a custom handler
    stream = io.StringIO()
    custom_handler = logging.StreamHandler(stream)
    add_log_handler(custom_handler)

    # Verify handler is added
    logger = logging.getLogger("litserve")
    assert custom_handler in logger.handlers

    # Test the handler works
    test_message = "Test handler message"
    logger.info(test_message)
    assert test_message in stream.getvalue()


@pytest.fixture(autouse=True)
def cleanup_logger():
    yield
    logger = logging.getLogger("litserve")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
