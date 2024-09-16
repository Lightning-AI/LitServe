import dataclasses
import logging
from abc import ABC
from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    pass


@dataclasses.dataclass
class EventTypes:
    LITAPI_SETUP_START = "on_litapi_setup_start"
    LITAPI_SETUP_END = "on_litapi_setup_end"
    LITAPI_DECODE_REQUEST_START = "on_litapi_decode_request_start"
    LITAPI_DECODE_REQUEST_END = "on_litapi_decode_request_end"
    LITAPI_ENCODE_RESPONSE_START = "on_litapi_encode_response_start"
    LITAPI_ENCODE_RESPONSE_END = "on_litapi_encode_response_end"
    LITAPI_PREDICT_START = "on_litapi_predict_start"
    LITAPI_PREDICT_END = "on_litapi_predict_end"
    ENDPOINT_START = "on_endpoint_start"
    ENDPOINT_END = "on_endpoint_end"


class Callback(ABC):
    def on_endpoint_start(self, *args, **kwargs):
        """Called before LitServer /predict endpoint is called."""

    def on_endpoint_end(self, *args, **kwargs):
        """Called after LitServer /predict endpoint is called."""

    def on_litapi_predict_start(self, *args, **kwargs):
        """Called before LitAPI.predict() is called."""

    def on_litapi_predict_end(self, *args, **kwargs):
        """Called after LitAPI.predict() is called."""

    def on_litapi_decode_request_start(self, *args, **kwargs):
        """Called before LitAPI.decode_request() is called."""

    def on_litapi_decode_request_end(self, *args, **kwargs):
        """Called after LitAPI.decode_request() is called."""

    def on_litapi_encode_response_start(self, *args, **kwargs):
        """Called before LitAPI.decode_request() is called."""

    def on_litapi_encode_response_end(self, *args, **kwargs):
        """Called after LitAPI.encode_response() is called."""

    def on_litapi_setup_start(self, *args, **kwargs):
        """Called before LitAPI.setup() is called."""

    def on_litapi_setup_end(self, *args, **kwargs):
        """Called after LitAPI.setup() is called."""


class CallbackRunner:
    def __init__(self):
        self._callbacks = []

    def add_callbacks(self, callbacks: Union[Callback, List[Callback]]):
        if isinstance(callbacks, list):
            self._callbacks.extend(callbacks)
        else:
            self._callbacks.append(callbacks)
        print("added runners")

    def trigger_event(self, event_name, *args, **kwargs):
        """Triggers an event, invoking all registered callbacks for that event."""
        for callback in self._callbacks:
            try:
                getattr(callback, event_name)(*args, **kwargs)
            except Exception:
                # Handle exceptions to prevent one callback from disrupting others
                logging.exception(f"Error in callback '{callback.name}' during event '{event_name}'")
