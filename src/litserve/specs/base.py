# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Generator, List, Optional, Union

if TYPE_CHECKING:
    from litserve import LitAPI, LitServer


class LitSpec:
    """Spec will have its own encode, and decode."""

    def __init__(self):
        self._endpoints = []
        self.api_path = None
        self._server: LitServer = None
        self._max_batch_size = 1
        self.response_buffer = None
        self.request_queue = None
        self.response_queue_id = None

    @property
    def stream(self):
        return False

    def pre_setup(self, lit_api: "LitAPI"):
        pass

    def setup(self, server: "LitServer"):
        """This method is called by the server to connect the spec to the server."""
        self.response_buffer = server.response_buffer
        self.request_queue = server._get_request_queue(self.api_path)
        self.data_streamer = server.data_streamer

    def add_endpoint(self, path: str, endpoint: Callable, methods: List[str]):
        """Register an endpoint in the spec."""
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints(self):
        return self._endpoints.copy()

    @abstractmethod
    def decode_request(self, request, meta_kwargs):
        """Convert the request payload to your model input."""
        pass

    @abstractmethod
    def encode_response(self, output, meta_kwargs):
        """Convert the model output to a response payload.

        To enable streaming, it should yield the output.

        """
        pass

    def as_async(self) -> "_AsyncSpecWrapper":
        return _AsyncSpecWrapper(self)


class _AsyncSpecWrapper:
    def __init__(self, spec: LitSpec):
        self._spec = spec

    def __getattr__(self, name):
        # Delegate all other attributes/methods to the wrapped spec
        return getattr(self._spec, name)

    async def decode_request(self, request, context_kwargs: Optional[dict] = None):
        return self._spec.decode_request(request, context_kwargs)

    async def encode_response(self, output: Union[Generator, AsyncGenerator], context_kwargs: Optional[dict] = None):
        return self._spec.encode_response(output, context_kwargs)
