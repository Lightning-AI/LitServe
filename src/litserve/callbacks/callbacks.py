import typing

if typing.TYPE_CHECKING:
    from litserve import LitServer, LitAPI


class Callback:
    def __init__(self):
        self.callbacks = []

    def on_endpoint_setup_start(self, server: "LitServer", lit_api: "LitAPI"):
        """Called before LitServer.setup() is called."""

    def on_endpoint_setup_end(self, server: "LitServer", lit_api: "LitAPI"):
        """Called after LitServer.setup() is called."""

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
