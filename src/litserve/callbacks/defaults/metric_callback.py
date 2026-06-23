import logging
import time
import typing

from litserve.callbacks.base import Callback

if typing.TYPE_CHECKING:
    from litserve import LitAPI

logger = logging.getLogger(__name__)


class PredictionTimeLogger(Callback):
    def on_before_predict(self, lit_api: "LitAPI"):
        self._start_time = time.perf_counter()

    def on_after_predict(self, lit_api: "LitAPI"):
        elapsed = time.perf_counter() - self._start_time
        logger.info(f"Prediction took {elapsed:.2f} seconds")


class RequestTracker(Callback):
    def on_request(self, active_requests: int, **kwargs):
        logger.info(f"Active requests: {active_requests}")
