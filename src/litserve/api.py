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
import inspect
import json
import warnings
from abc import ABC, abstractmethod
from typing import Optional
from queue import Queue

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
    def setup(self, devices):
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

    @abstractmethod
    def predict(self, x, **kwargs):
        """Run the model on the input and return or yield the output."""
        pass

    def _unbatch_no_stream(self, output):
        if isinstance(output, str):
            warnings.warn(
                "The 'predict' method returned a string instead of a list of predictions. "
                "When batching is enabled, 'predict' must return a list to handle multiple inputs correctly. "
                "Please update the 'predict' method to return a list of predictions to avoid unexpected behavior.",
                UserWarning
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

    def _sanitize(self, max_batch_size: int, spec: Optional[LitSpec]):
        if self.stream:
            self._default_unbatch = self._unbatch_stream
        else:
            self._default_unbatch = self._unbatch_no_stream

        # we will sanitize regularly if no spec
        # in case, we have spec then:
        # case 1: spec implements a streaming API
        # Case 2: spec implements a non-streaming API
        if spec:
            # TODO: Implement sanitization
            self._spec = spec
            return

        original = self.unbatch.__code__ is LitAPI.unbatch.__code__
        if (
            self.stream
            and max_batch_size > 1
            and not all([
                inspect.isgeneratorfunction(self.predict),
                inspect.isgeneratorfunction(self.encode_response),
                (original or inspect.isgeneratorfunction(self.unbatch)),
            ])
        ):
            raise ValueError(
                """When `stream=True` with max_batch_size > 1, `lit_api.predict`, `lit_api.encode_response` and
                `lit_api.unbatch` must generate values using `yield`.

             Example:

                def predict(self, inputs):
                    ...
                    for i in range(max_token_length):
                        yield prediction

                def encode_response(self, outputs):
                    for output in outputs:
                        encoded_output = ...
                        yield encoded_output

                def unbatch(self, outputs):
                    for output in outputs:
                        unbatched_output = ...
                        yield unbatched_output
             """
            )

        if self.stream and not all([
            inspect.isgeneratorfunction(self.predict),
            inspect.isgeneratorfunction(self.encode_response),
        ]):
            raise ValueError(
                """When `stream=True` both `lit_api.predict` and
             `lit_api.encode_response` must generate values using `yield`.

             Example:

                def predict(self, inputs):
                    ...
                    for i in range(max_token_length):
                        yield prediction

                def encode_response(self, outputs):
                    for output in outputs:
                        encoded_output = ...
                        yield encoded_output
             """
            )

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
