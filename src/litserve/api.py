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
from abc import ABC, abstractmethod

from pydantic import BaseModel

from litserve.specs.base import LitSpec


def no_batch_unbatch_message_no_stream(obj, data):
    return f"""
        You set `max_batch_size > 1`, but the default implementation for batch() and unbatch() only supports
        PyTorch tensors or NumPy ndarrays, while we found {type(data)}.
        Please implement these two methods in {obj.__class__.__name__}.

        Example:

        def batch(self, inputs):
            return np.stack(inputs)

        def unbatch(self, output):
            return list(output)
    """


def no_batch_unbatch_message_stream(obj, data):
    return f"""
        You set `max_batch_size > 1`, but the default implementation for batch() and unbatch() only supports
        PyTorch tensors or NumPy ndarrays, while we found {type(data)}.
        Please implement these two methods in {obj.__class__.__name__}.

        Example:

        def batch(self, inputs):
            return np.stack(inputs)

        def unbatch(self, output):
            for out in output:
                yield list(out)
    """


class LitAPI(ABC):
    _stream: bool = False
    _default_unbatch: callable = None
    _spec: LitSpec = None

    @abstractmethod
    def setup(self, devices):
        """Setup the model so it can be called in `predict`."""
        pass

    def decode_request(self, request):
        """Convert the request payload to your model input."""
        if self._spec:
            return self._spec.decode_request(request)
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

        if self.stream:
            message = no_batch_unbatch_message_stream(self, inputs)
        else:
            message = no_batch_unbatch_message_no_stream(self, inputs)
        raise NotImplementedError(message)

    @abstractmethod
    def predict(self, x):
        """Run the model on the input and return the output."""
        pass

    def _unbatch_no_stream(self, output):
        if hasattr(output, "__torch_function__") or output.__class__.__name__ == "ndarray":
            return list(output)
        message = no_batch_unbatch_message_no_stream(self, output)
        raise NotImplementedError(message)

    def _unbatch_stream(self, output_stream):
        for output in output_stream:
            if hasattr(output, "__torch_function__") or output.__class__.__name__ == "ndarray":
                yield list(output)
            else:
                message = no_batch_unbatch_message_no_stream(self, output)
                raise NotImplementedError(message)

    def unbatch(self, output):
        """Convert a batched output to a list of outputs."""
        return self._default_unbatch(output)

    def encode_response(self, output):
        """Convert the model output to a response payload.

        It should return the output.

        """
        if self._spec:
            return self._spec.encode_response(output)
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

    def sanitize(self, max_batch_size: int, spec: LitSpec):
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
