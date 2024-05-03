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
        if self.decode_request_hook:
            return self.decode_request_hook(request)
        return self._lit_api.decode_request(request)

    def encode_response(self, output):
        if self.encode_response_hook:
            return self.encode_response_hook(output)

        return self._lit_api.encode_response(output)
