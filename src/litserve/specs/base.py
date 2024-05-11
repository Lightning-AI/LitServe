from typing import Callable, TYPE_CHECKING
from litserve.api import LitAPI

if TYPE_CHECKING:
    from litserve import LitServer


class LitSpec(LitAPI):
    """Spec will have its own encode, and decode.

    - if not then fallback to litapi
    We have to call `predict` from LitAPI

    """

    def __init__(self):
        self._endpoints = []

        self._server: "LitServer" = None
        self._lit_api = None

    def setup(self, server: "LitServer"):
        self._server = server

    def add_endpoint(self, path: str, endpoint: Callable, methods: list[str]):
        """Register an endpoint in the spec."""
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints(self):
        return self._endpoints.copy()

    def decode_request(self, *args, **kwargs):
        return self.lit_api.decode_request(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.lit_api.predict(*args, **kwargs)

    def encode_response(self, *args, **kwargs):
        return self.lit_api.encode_response(*args, **kwargs)

    @property
    def lit_api(self) -> LitAPI:
        return self._lit_api
