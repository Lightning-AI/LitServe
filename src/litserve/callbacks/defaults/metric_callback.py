import time
import typing
from logging import getLogger

from litserve.callbacks.base import Callback

if typing.TYPE_CHECKING:
    from litserve import LitAPI

logger = getLogger(__name__)


class PredictionTimeLogger(Callback):
    def on_before_predict(self, lit_api: "LitAPI"):
        t0 = time.perf_counter()
        self._start_time = t0

    def on_after_predict(self, lit_api: "LitAPI"):
        t1 = time.perf_counter()
        elapsed = t1 - self._start_time
        print(f"Prediction took {elapsed:.2f} seconds", flush=True)


class RequestTracker(Callback):
    def on_request(self, active_requests: int, **kwargs):
        print(f"Active requests: {active_requests}", flush=True)
