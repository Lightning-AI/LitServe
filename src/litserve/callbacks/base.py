import logging
from abc import ABC
from enum import Enum
from typing import List, Union

logger = logging.getLogger(__name__)


class EventTypes(Enum):
    BEFORE_SETUP = "on_before_setup"
    AFTER_SETUP = "on_after_setup"
    BEFORE_DECODE_REQUEST = "on_before_decode_request"
    AFTER_DECODE_REQUEST = "on_after_decode_request"
    BEFORE_ENCODE_RESPONSE = "on_before_encode_response"
    AFTER_ENCODE_RESPONSE = "on_after_encode_response"
    BEFORE_PREDICT = "on_before_predict"
    AFTER_PREDICT = "on_after_predict"
    ON_SERVER_START = "on_server_start"
    ON_SERVER_END = "on_server_end"
    ON_REQUEST = "on_request"
    ON_RESPONSE = "on_response"


class Callback(ABC):
    def on_before_setup(self, *args, **kwargs):
        """Called before setup is started."""

    def on_after_setup(self, *args, **kwargs):
        """Called after setup is completed."""

    def on_before_decode_request(self, *args, **kwargs):
        """Called before request decoding is started."""

    def on_after_decode_request(self, *args, **kwargs):
        """Called after request decoding is completed."""

    def on_before_encode_response(self, *args, **kwargs):
        """Called before response encoding is started."""

    def on_after_encode_response(self, *args, **kwargs):
        """Called after response encoding is completed."""

    def on_before_predict(self, *args, **kwargs):
        """Called before prediction is started."""

    def on_after_predict(self, *args, **kwargs):
        """Called after prediction is completed."""

    def on_server_start(self, *args, **kwargs):
        """Called before server starts."""

    def on_server_end(self, *args, **kwargs):
        """Called when server terminates."""

    def on_request(self, *args, **kwargs):
        """Called when request enters the endpoint function."""

    def on_response(self, *args, **kwargs):
        """Called when response is generated from the worker and ready to return to the client."""

    # Adding a new hook? Register it with the EventTypes dataclass too,


class CallbackRunner:
    def __init__(self, callbacks: Union[Callback, List[Callback]] = None):
        self._callbacks = []
        if callbacks:
            self._add_callbacks(callbacks)

    def _add_callbacks(self, callbacks: Union[Callback, List[Callback]]):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise ValueError(f"Invalid callback type: {callback}")
        self._callbacks.extend(callbacks)

    def trigger_event(self, event_name, *args, **kwargs):
        """Triggers an event, invoking all registered callbacks for that event."""
        for callback in self._callbacks:
            try:
                getattr(callback, event_name)(*args, **kwargs)
            except Exception:
                # Handle exceptions to prevent one callback from disrupting others
                logger.exception(f"Error in callback '{callback}' during event '{event_name}'")


class NoopCallback(Callback):
    """This callback does nothing."""
