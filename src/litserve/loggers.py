import functools
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Optional, Union, TYPE_CHECKING

from starlette.types import ASGIApp

if TYPE_CHECKING:
    from litserve import LitServer


class Logger(ABC):
    def __init__(self):
        self._connector = None  # Reference to _LoggerConnector
        self._config = {}

    def set_connector(self, connector: "_LoggerConnector"):
        self._connector = connector

    def mount(self, path: str, app: ASGIApp) -> None:
        """Mount an ASGI app endpoint to LitServer. Use this method when you want to add an additional endpoint to the
        server such as /metrics endpoint for prometheus metrics.

        Args:
            path (str): The path to mount the app to.
            app (ASGIApp): The ASGI app to mount.

        """
        self._config.update({"mount": {"path": path, "app": app}})

    @abstractmethod
    def process(self, key, value):
        """Process a log entry from the log queue.

        This method should be implemented to define the specific logic for processing
        log entries.

        Args:
            key (str): The key associated with the log entry, typically indicating the type or category of the log.
            value (Any): The value associated with the log entry, containing the actual log data.

        Raises:
            NotImplementedError: This method must be overridden by subclasses. If not, calling this method will raise
            a NotImplementedError.

        Example:
            Here is an example of a Logger that logs monitoring metrics using Prometheus:

            from prometheus_client import Counter

            class PrometheusLogger(Logger):
                def __init__(self):
                    super().__init__()
                    self._metric_counter = Counter('log_entries', 'Count of log entries')

                def process(self, key, value):
                    # Increment the Prometheus counter for each log entry
                    self._metric_counter.inc()
                    print(f"Logged {key}: {value}")

        """
        raise NotImplementedError


class _LoggerConnector:
    """_LoggerConnector is responsible for connecting Logger instances with the LitServer and managing their lifecycle.

    This class handles the following tasks:
    - Manages a queue (multiprocessing.Queue) where log data is placed using the LitAPI.log method.
    - Initiates a separate process to consume the log queue and process the log data using the associated
    Logger instances.

    """

    def __init__(self, lit_server: "LitServer", loggers: Optional[Union[List[Logger], Logger]] = None):
        self._loggers = []
        self._lit_server = lit_server
        if loggers is None:
            return  # No loggers to add
        if isinstance(loggers, list):
            for logger in loggers:
                if not isinstance(logger, Logger):
                    raise ValueError("Logger must be an instance of litserve.Logger")
                self.add_logger(logger)
        elif isinstance(loggers, Logger):
            self.add_logger(loggers)
        else:
            raise ValueError("loggers must be a list or an instance of litserve.Logger")

    def _mount(self, path: str, app: ASGIApp) -> None:
        self._lit_server.app.mount(path, app)

    def add_logger(self, logger: Logger):
        self._loggers.append(logger)
        logger.set_connector(self)  # Set the connector reference in Logger
        if "mount" in logger._config:
            self._mount(logger._config["mount"]["path"], logger._config["mount"]["app"])

    @staticmethod
    def _process_logger_queue(loggers: List[Logger], queue):
        while True:
            key, value = queue.get()
            for logger in loggers:
                logger.process(key, value)

    @functools.cache  # Run once per LitServer instance
    def run(self):
        ctx = mp.get_context("spawn")
        queue = self._lit_server.logger_queue
        # Disconnect the logger connector from the LitServer to avoid pickling issues
        self._lit_server = None
        if not self._loggers:
            return

        process = ctx.Process(
            target=_LoggerConnector._process_logger_queue,
            args=(
                self._loggers,
                queue,
            ),
        )
        process.start()
