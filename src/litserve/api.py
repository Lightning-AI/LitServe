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
from abc import ABC, abstractmethod


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

    @abstractmethod
    def setup(self, devices):
        """Setup the model so it can be called in `predict`."""
        pass

    @abstractmethod
    def decode_request(self, request):
        """Convert the request payload to your model input."""
        pass

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
        raise NotImplementedError(no_batch_unbatch_message_no_stream(self, inputs))

    @abstractmethod
    def predict(self, x):
        """Run the model on the input and return or yield the output."""
        pass

    def unbatch(self, output):
        """Convert a batched output to a list of outputs."""
        if hasattr(output, "__torch_function__") or output.__class__.__name__ == "ndarray":
            if self._stream:
                yield from list(output)
            else:
                return list(output)

        if self.stream:
            message = no_batch_unbatch_message_stream(self, output)
        else:
            message = no_batch_unbatch_message_no_stream(self, output)
        raise NotImplementedError(message)

    @abstractmethod
    def encode_response(self, output):
        """Convert the model output to a response payload.

        To enable streaming, it should yield the output.

        """
        pass

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = value

    def sanitize(self, max_batch_size: int):
        if (
            self.stream
            and max_batch_size > 1
            and not all([
                inspect.isgeneratorfunction(self.predict),
                inspect.isgeneratorfunction(self.encode_response),
                inspect.isgeneratorfunction(self.unbatch),
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
