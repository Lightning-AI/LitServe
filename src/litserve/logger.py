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
        raise NotImplementedError


class _LoggerConnector:
    """self.log will put data into a Queue multiple workers will put in the same queue The logger will have a single
    instance which would process the consumed data."""

    def __init__(self, lit_server: "LitServer", loggers: Optional[Union[List[Logger], Logger]] = None):
        self._loggers = []
        self._lit_server = lit_server
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

    def _process_logger_queue(self):
        queue = self._lit_server.log_queue
        while True:
            key, value = queue.get()
            for logger in self._loggers:
                logger.process(key, value)
