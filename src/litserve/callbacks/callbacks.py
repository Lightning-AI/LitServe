from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from litserve import LitServer, LitAPI


class Callback:
    def on_endpoint_start(self, server: "LitServer", lit_api: "LitAPI"):
        """Called before LitServer /predict endpoint is called."""

    def on_endpoint_end(self, server: "LitServer", lit_api: "LitAPI"):
        """Called after LitServer /predict endpoint is called."""

    def on_litapi_predict_start(self, server: "LitServer", lit_api: "LitAPI"):
        """Called before LitAPI.predict() is called."""

    def on_litapi_predict_end(self, server: "LitServer", lit_api: "LitAPI"):
        """Called after LitAPI.predict() is called."""

    def on_litapi_decode_request_start(self, server: "LitServer", lit_api: "LitAPI"):
        """Called before LitAPI.decode_request() is called."""

    def on_litapi_decode_request_end(self, server: "LitServer", lit_api: "LitAPI"):
        """Called after LitAPI.decode_request() is called."""

    def on_litapi_encode_response_start(self, server: "LitServer", lit_api: "LitAPI"):
        """Called before LitAPI.decode_request() is called."""

    def on_litapi_encode_response_end(self, server: "LitServer", lit_api: "LitAPI"):
        """Called after LitAPI.encode_response() is called."""

    def on_litapi_setup_start(self, server: "LitServer", lit_api: "LitAPI"):
        """Called before LitAPI.setup() is called."""

    def on_litapi_setup_end(self, server: "LitServer", lit_api: "LitAPI"):
        """Called after LitAPI.setup() is called."""


class _CallbackConnector:
    def __init__(self, server: "LitServer"):
        self._server = server
        self._callbacks = []

    def add_callbacks(self, callbacks: Union[Callback, List[Callback]]):
        if isinstance(callbacks, list):
            self._callbacks.extend(callbacks)
        else:
            self._callbacks.append(callbacks)
