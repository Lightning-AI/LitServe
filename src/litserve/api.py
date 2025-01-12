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
import json
import warnings
from abc import ABC, abstractmethod
from queue import Queue
from typing import Optional

from pydantic import BaseModel

from litserve.specs.base import LitSpec


class LitAPI(ABC):
    _stream: bool = False
    _default_unbatch: callable = None
    _spec: LitSpec = None
    _device: Optional[str] = None
    _logger_queue: Optional[Queue] = None
    request_timeout: Optional[float] = None

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

    def pre_setup(self, max_batch_size: int, spec: Optional[LitSpec]):
        self.max_batch_size = max_batch_size
        if self.stream:
            self._default_unbatch = self._unbatch_stream
        else:
            self._default_unbatch = self._unbatch_no_stream

        if spec:
            self._spec = spec
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
