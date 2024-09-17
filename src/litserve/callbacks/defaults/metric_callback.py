import time
import typing
from logging import getLogger

from ..base import Callback

if typing.TYPE_CHECKING:
    from litserve import LitAPI

logger = getLogger(__name__)


class PredictionTimeLogger(Callback):
    def on_litapi_predict_start(self, lit_api: "LitAPI"):
        t0 = time.perf_counter()
        self._start_time = t0

    def on_litapi_predict_end(self, lit_api: "LitAPI"):
        t1 = time.perf_counter()
        elapsed = t1 - self._start_time
        print(f"Prediction took {elapsed:.2f} seconds", flush=True)
