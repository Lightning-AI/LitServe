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
import asyncio
import inspect
import json
import warnings
from abc import ABC, abstractmethod
from queue import Queue
from typing import Callable, Optional

from pydantic import BaseModel

from litserve.specs.base import LitSpec


class LitAPI(ABC):
    _stream: bool = False
    _default_unbatch: Optional[Callable] = None
    _spec: Optional[LitSpec] = None
    _device: Optional[str] = None
    _logger_queue: Optional[Queue] = None
    request_timeout: Optional[float] = None

    def __init__(self, max_batch_size: int = 1, batch_timeout: float = 0.0, enable_async: bool = False):
        """Initialize a LitAPI instance.

        Args:
            max_batch_size: Maximum number of requests to process in a batch.
            batch_timeout: Maximum time to wait for a batch to fill before processing.
            enable_async: Enable async support.

        """

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")

        if batch_timeout < 0:
            raise ValueError("batch_timeout must be greater than or equal to 0")
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.enable_async = enable_async
        self._validate_async_methods()

    def _validate_async_methods(self):
        """Validate that async methods are properly implemented when enable_async is True."""
        if self.enable_async:
            # check if LitAPI methods are coroutines or async generators
            for method in ["decode_request", "predict", "encode_response"]:
                method_obj = getattr(self, method)
                if not (asyncio.iscoroutinefunction(method_obj) or inspect.isasyncgenfunction(method_obj)):
                    raise ValueError("""LitAPI(enable_async=True) requires all methods to be coroutines.

Please either set enable_async=False or implement the following methods as coroutines:
Example:
    class MyLitAPI(LitAPI):
        async def decode_request(self, request, **kwargs):
            return request
        async def predict(self, x, **kwargs):
            return x
        async def encode_response(self, output, **kwargs):
            return output

Streaming example:
    class MyStreamingAPI(LitAPI):
        async def predict(self, x, **kwargs):
            for i in range(10):
                await asyncio.sleep(0.1)  # simulate async work
                yield f"Token {i}: {x}"
""")

    @abstractmethod
    def setup(self, device):
        """Setup the model so it can be called in `predict`."""
        pass

    def decode_request(self, request, **kwargs):
        """Convert the request payload to your model input."""
        if self._spec:
            return self._spec.decode_request(request, **kwargs)
        return request

    def batch(self, inputs):
        """Convert a list of inputs to a batched input."""
        # consider assigning an implementation when starting server
        # to avoid the runtime cost of checking (should be negligible)
        if hasattr(inputs[0], "__torch_function__"):
            import torch

            return torch.stack(inputs)
        if inputs[0].__class__.__name__ == "ndarray":
            import numpy

            return numpy.stack(inputs)

        return inputs

    def predict(self, x, **kwargs):
        """Run the model on the input and return or yield the output."""
        raise NotImplementedError("predict is not implemented")

    def _unbatch_no_stream(self, output):
        if isinstance(output, str):
            warnings.warn(
                "The 'predict' method returned a string instead of a list of predictions. "
                "When batching is enabled, 'predict' must return a list to handle multiple inputs correctly. "
                "Please update the 'predict' method to return a list of predictions to avoid unexpected behavior.",
                UserWarning,
            )
        return list(output)

    def _unbatch_stream(self, output_stream):
        for output in output_stream:
            yield list(output)

    def unbatch(self, output):
        """Convert a batched output to a list of outputs."""
        if self._default_unbatch is None:
            raise ValueError(
                "Default implementation for `LitAPI.unbatch` method was not found. "
                "Please implement the `LitAPI.unbatch` method."
            )
        return self._default_unbatch(output)

    def encode_response(self, output, **kwargs):
        """Convert the model output to a response payload.

        To enable streaming, it should yield the output.

        """
        if self._spec:
            return self._spec.encode_response(output, **kwargs)
        return output

    def format_encoded_response(self, data):
        if isinstance(data, dict):
            return json.dumps(data) + "\n"
        if isinstance(data, BaseModel):
            return data.model_dump_json() + "\n"
        return data

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def pre_setup(self, spec: Optional[LitSpec]):
        if self.batch_timeout > self.request_timeout and self.request_timeout not in (False, -1):
            raise ValueError("batch_timeout must be less than request_timeout")

        if self.stream:
            self._default_unbatch = self._unbatch_stream
        else:
            self._default_unbatch = self._unbatch_no_stream

        if spec:
            self._spec = spec
            spec._max_batch_size = self.max_batch_size
            spec.pre_setup(self)

    def set_logger_queue(self, queue: Queue):
        """Set the queue for logging events."""

        self._logger_queue = queue

    def log(self, key, value):
        """Log a key-value pair to the server."""
        if self._logger_queue is None:
            warnings.warn(
                f"Logging event ('{key}', '{value}') attempted without a configured logger. "
                "To track and visualize metrics, please initialize and attach a logger. "
                "If this is intentional, you can safely ignore this message."
            )
            return
        self._logger_queue.put((key, value))

    def has_active_requests(self) -> bool:
        raise NotImplementedError("has_active_requests is not implemented")

    def has_capacity(self) -> bool:
        raise NotImplementedError("has_capacity is not implemented")

    def health(self) -> bool:
        """Check the additional health status of the API.

        This method is used in the /health endpoint of the server to determine the health status.
        Users can extend this method to include additional health checks specific to their application.

        Returns:
            bool: True if the API is healthy, False otherwise.

        """
        return True
