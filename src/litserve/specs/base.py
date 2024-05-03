from typing import Callable
from litserve.api import LitAPI


class LitSpec:
    """Spec will have its own encode, and decode.

    - if not then fallback to litapi
    We have to call `predict` from LitAPI

    """

    _endpoints = []

    decode_request_hook = None
    encode_response_hook = None
    _server = None
    _lit_api: LitAPI = None

    def setup(self, obj):
        raise NotImplementedError()

    def add_endpoint(self, path: str, endpoint: Callable, methods: list[str]):
        """Register an endpoint in the spec."""
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints(self):
        return self._endpoints.copy()

    def decode_request(self, request):
        # TODO: LitAPI.decode_request is abstractmethod and users will be forced to implement that.
        if self.decode_request_hook and not callable(self.decode_request_hook):
            raise ValueError(
                "decode_request_hook was defined but is not callable. It must be a callable "
                "function/method to replace LitAPI.decode_request."
            )
        return self.decode_request_hook(request) if self.decode_request_hook else self._lit_api.decode_request(request)

    def encode_response(self, output):
        if self.encode_response_hook and not callable(self.encode_response_hook):
            raise ValueError(
                "encode_response_hook was defined but is not callable. It must be a callable "
                "function/method to replace LitAPI.encode_response."
            )
        return self.encode_response_hook(output) if self.encode_response_hook else self._lit_api.encode_response(output)
