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
from typing import Optional

from pydantic import BaseModel

from litserve.specs.base import LitSpec


class LitAPI(ABC):
    _stream: bool = False
    _default_unbatch: callable = None
    _spec: LitSpec = None
    _device: Optional[str] = None
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

    def preprocess(self, x, **kwargs):
        """Preprocess the input data before passing it to the model for inference.

        The `preprocess` function handles necessary transformations (e.g., data normalization,
        tokenization, feature extraction, or image resizing) before sending the data to
        the model for prediction.

        Args:
            x: Input data, either a single instance or a batch, depending on the model’s requirements.
            kwargs: Additional arguments for specific preprocessing tasks.

        Returns:
            Preprocessed data in a format compatible with the model's `predict` function.

        Usage:
            - Separate Workers for Preprocessing and Inference: If the preprocessing step is
              computationally intensive, it is run on separate process workers to prevent it from
              blocking the main prediction flow. The processed data is passed via a queue to the
              inference workers, ensuring both stages can work in parallel.
            - Performance Optimization: By decoupling preprocessing and inference, the system
              can handle more requests simultaneously, reducing latency and improving throughput.
              For example, while one request is being preprocessed, another can be inferred,
              overlapping the time spent on both operations.

        Example:
            Consider batch_size = 1, with 3 requests, and 1 inference worker:
            Preprocessing takes 4s and Inference takes 2s.

        1. Without Separate Preprocessing Workers (Sequential):
            Request 1 → Preprocess → Inference
            Request 2 → Preprocess → Inference
            Request 3 → Preprocess → Inference

                Request 1: |-- Preprocess --|-- Inference --|
                Request 2:                                  |-- Preprocess --|-- Inference --|
                Request 3:                                                                   |-- Preprocess --|-- Inference --|


            Total time: (4s + 2s) * 3 = 18s

        2. With Separate Preprocessing Workers (Concurrent):
            Request 1 → Preprocess → Inference
            Request 2 → Preprocess → Inference
            Request 3 → Preprocess → Inference

            Request 1: |-- Preprocess --|-- Inference --|
            Request 2:                  |-- Preprocess --|-- Inference --|
            Request 3:                                    |-- Preprocess --|-- Inference --|

            Total time: 4s + 4s + 4s + 2s = 14s

        When to Override:
            - When preprocessing is time-consuming: If your preprocessing step involves heavy
              computations (e.g., applying complex filters, large-scale image processing, or
              extensive feature extraction), you should override `preprocess` to run it separately
              from inference. This is especially important when preprocessing and inference both
              take considerable time, as overlapping the two processes improves throughput.

            - If both preprocessing and inference take significant time (e.g., several
              seconds), running them concurrently can significantly reduce latency and improve
              performance. For example, in high-latency models like image segmentation or NLP
              models that require tokenization, separating the two stages will be highly effective.

            - Less effective for fast models: If both preprocessing and inference take only a
              few milliseconds each, the benefit of separating them into parallel processes may
              be minimal. In such cases, the overhead of managing multiple workers and queues may
              outweigh the performance gain.

            - Dynamic workloads: If your workload fluctuates or you expect periods of high
              demand, decoupling preprocessing from inference allows you to scale each stage
              independently by adding more workers based on the current system load.

        """
        pass

    @abstractmethod
    def predict(self, x, **kwargs):
        """Run the model on the input and return or yield the output."""
        pass

    def _unbatch_no_stream(self, output):
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
