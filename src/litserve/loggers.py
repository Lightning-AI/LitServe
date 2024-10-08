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
import functools
import logging
import multiprocessing as mp
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

from starlette.types import ASGIApp

module_logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from litserve import LitServer


class Logger(ABC):
    def __init__(self):
        self._config = {}

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
        raise NotImplementedError  # pragma: no cover


class _LoggerProxy:
    def __init__(self, logger_class):
        self.logger_class = logger_class

    def create_logger(self):
        return self.logger_class()


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
        if "mount" in logger._config:
            self._mount(logger._config["mount"]["path"], logger._config["mount"]["app"])

    @staticmethod
    def _is_picklable(obj):
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, TypeError, AttributeError):
            module_logger.warning(f"Logger {obj.__class__.__name__} is not pickleable and might not work properly.")
            return False

    @staticmethod
    def _process_logger_queue(logger_proxies: List[_LoggerProxy], queue):
        loggers = [proxy if isinstance(proxy, Logger) else proxy.create_logger() for proxy in logger_proxies]
        while True:
            key, value = queue.get()
            for logger in loggers:
                try:
                    logger.process(key, value)
                except Exception as e:
                    module_logger.error(
                        f"{logger.__class__.__name__} ran into an error while processing log for entry "
                        f"with key {key} and value {value}: {e}"
                    )

    @functools.cache  # Run once per LitServer instance
    def run(self, lit_server: "LitServer"):
        queue = lit_server.logger_queue
        lit_server.lit_api.set_logger_queue(queue)

        # Disconnect the logger connector from the LitServer to avoid pickling issues
        self._lit_server = None

        if not self._loggers:
            return

        # Create proxies for loggers
        logger_proxies = []
        for logger in self._loggers:
            if self._is_picklable(logger):
                logger_proxies.append(logger)
            else:
                logger_proxies.append(_LoggerProxy(logger.__class__))

        module_logger.debug(f"Starting logger process with {len(logger_proxies)} loggers")
        ctx = mp.get_context("spawn")
        process = ctx.Process(
            target=_LoggerConnector._process_logger_queue,
            args=(
                logger_proxies,
                queue,
            ),
        )
        process.start()
