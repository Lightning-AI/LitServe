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
from abc import ABC
from collections.abc import Awaitable, Callable
from queue import Queue
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel

from litserve.specs.base import LitSpec
from litserve.utils import _TimedInitMeta

if TYPE_CHECKING:
    from litserve.loops.base import LitLoop
    from litserve.mcp import MCP


class LitAPI(ABC, metaclass=_TimedInitMeta):
    """Define inference logic for the model.

    LitAPI is the core abstraction for serving AI models with LitServe. It provides a clean
    interface for model loading, request processing, and response generation with automatic
    optimizations like batching, streaming, and async processing.

    Core Workflow:
        1. **setup()**: Load and initialize the model once per worker
        2. **decode_request()**: Convert HTTP request to model input format
        3. **predict()**: Run model inference on the input
        4. **encode_response()**: Convert model output to HTTP response format

    Quick Start:
        ```python
        import litserve as ls

        class MyAPI(ls.LitAPI):
            def setup(self, device):
                self.model = lambda x: x**2

            def predict(self, x):
                return self.model(x["input"])

        server = ls.LitServer(MyAPI())
        server.run()
        ```

    Required Methods:
        setup(device): Initialize the model and resources
        predict(x): Core inference logic

    Optional Methods:
        decode_request(request): Transform HTTP requests to model input
        encode_response(output): Transform model outputs to HTTP responses
        batch(inputs)/unbatch(outputs): Custom batching logic

    Configuration:
        max_batch_size: Batch multiple requests for better GPU utilization. Defaults to 1.
        batch_timeout: Wait time for batch to fill (seconds). Defaults to 0.0.
        stream: Enable streaming responses for real-time output. Defaults to False.
        api_path: URL endpoint path. Defaults to "/predict".
        enable_async: Enable async/await for non-blocking operations. Defaults to False.
        spec: API specification (e.g., OpenAISpec for OpenAI compatibility). Defaults to None.
        mcp: Model Context Protocol integration for AI assistants. Defaults to None.

    Examples:
        Batched GPU Inference:
        ```python
        class BatchedAPI(ls.LitAPI):
            def setup(self, device):
                self.model = load_model().to(device)

            def predict(self, batch):
                return self.model(batch)

        api = BatchedAPI(max_batch_size=8, batch_timeout=0.1)
        ```

        Streaming LLM:
        ```python
        class StreamingLLM(ls.LitAPI):
            def setup(self, device):
                self.model = load_llm()

            def predict(self, prompt):
                for token in self.model.generate_stream(prompt):
                    yield token

        api = StreamingLLM(stream=True)
        ```

        OpenAI-Compatible:
        ```python
        from litserve.specs import OpenAISpec

        class ChatAPI(ls.LitAPI):
            def setup(self, device):
                self.model = load_chat_model()

            def predict(self, messages):
                return self.model.chat(messages)

        api = ChatAPI(spec=OpenAISpec())
        ```

    Performance Tips:
        - Use batching for GPU models to maximize utilization
        - Enable streaming for operations taking >1 second
        - Use async for I/O-bound operations (databases, external APIs)
        - Load models in setup(), not __init__
        - Monitor GPU memory usage with larger batch sizes

    See Also:
        - LitServer: Server class for hosting APIs
        - LitSpec: API specifications for standard interfaces

    """

    _stream: bool = False
    _default_unbatch: Optional[Callable] = None
    _spec: Optional[LitSpec] = None
    _device: Optional[str] = None
    _logger_queue: Optional[Queue] = None
    request_timeout: Optional[float] = None

    def __init__(
        self,
        max_batch_size: int = 1,
        batch_timeout: float = 0.0,
        api_path: str = "/predict",
        stream: bool = False,
        loop: Optional[Union[str, "LitLoop"]] = "auto",
        spec: Optional[LitSpec] = None,
        mcp: Optional["MCP"] = None,
        enable_async: bool = False,
    ):
        """Initialize LitAPI with configuration options."""

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")

        if batch_timeout < 0:
            raise ValueError("batch_timeout must be greater than or equal to 0")

        if isinstance(spec, LitSpec):
            stream = spec.stream

        if loop is None:
            loop = "auto"

        if isinstance(loop, str) and loop != "auto":
            raise ValueError("loop must be an instance of _BaseLoop or 'auto'")

        if not api_path.startswith("/"):
            raise ValueError(
                "api_path must start with '/'. "
                "Please provide a valid api path like '/predict', '/classify', or '/v1/predict'"
            )

        # Check if the batch and unbatch methods are overridden in the lit_api instance
        batch_overridden = self.batch.__code__ is not LitAPI.batch.__code__
        unbatch_overridden = self.unbatch.__code__ is not LitAPI.unbatch.__code__

        if batch_overridden and unbatch_overridden and max_batch_size == 1:
            warnings.warn(
                "The LitServer has both batch and unbatch methods implemented, "
                "but the max_batch_size parameter was not set."
            )

        self._api_path = api_path
        self.stream = stream
        self._loop = loop
        self._spec = spec
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.enable_async = enable_async
        self._validate_async_methods()
        self.mcp = mcp
        if mcp:
            mcp._connect(self)

    def _validate_async_methods(self):
        """Validate that async methods are properly implemented when enable_async is True."""
        if not self.enable_async:
            return

        # Define validation rules for each method
        validation_rules = {
            "decode_request": {
                "required_types": [asyncio.iscoroutinefunction, inspect.isasyncgenfunction],
                "error_type": "warning",
                "message": "should be an async function or async generator when enable_async=True",
            },
            "encode_response": {
                "required_types": [asyncio.iscoroutinefunction, inspect.isasyncgenfunction],
                "error_type": "warning",
                "message": "should be an async function or async generator when enable_async=True",
            },
            "predict": {
                "required_types": [inspect.isasyncgenfunction, asyncio.iscoroutinefunction],
                "error_type": "error",
                "message": "must be an async generator or async function when enable_async=True",
            },
        }

        errors = []
        warnings_list = []

        for method_name, rules in validation_rules.items():
            method_obj = getattr(self, method_name)

            # Check if method satisfies any of the required types
            is_valid = any(check_func(method_obj) for check_func in rules["required_types"])

            if not is_valid:
                message = f"{method_name} {rules['message']}"

                if rules["error_type"] == "error":
                    errors.append(message)
                else:
                    warnings_list.append(message)

        # Emit warnings
        for warning_msg in warnings_list:
            warnings.warn(f"{warning_msg}. LitServe will asyncify the method.", UserWarning)

        # Raise errors if any
        if errors:
            error_msg = "Async validation failed:\n" + "\n".join(f"- {err}" for err in errors)
            raise ValueError(error_msg)

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
        """Run the model on the input and return or yield the output.

        When batching is enabled (max_batch_size > 1), this method receives
        a batched input and must return a list-like structure where each element
        corresponds to one input in the batch.

        Returns:
            For non-batched mode: Single prediction output
            For batched mode: List, tuple, or array with one output per input

        """
        raise NotImplementedError("predict is not implemented")

    def _unbatch_no_stream(self, output):
        if isinstance(output, str):
            warnings.warn(
                "The 'predict' method returned a string instead of a list of predictions. "
                "When batching is enabled, 'predict' must return a list to handle multiple inputs correctly. "
                "Please update the 'predict' method to return a list of predictions to avoid unexpected behavior.",
                UserWarning,
            )
        elif isinstance(output, dict):
            warnings.warn(
                "The 'predict' method returned a dict instead of a list of predictions. "
                "When batching is enabled, 'predict' must return a list to handle multiple inputs correctly. "
                "For example, return [{'class_A': 0.2, 'class_B': 0.8}, {'class_A': 0.5, 'class_B': 0.5}] "
                "instead of {'class_A': [0.2, 0.5], 'class_B': [0.8, 0.5]}. "
                "Please update the 'predict' method to return a list of predictions to avoid unexpected behavior.",
                UserWarning,
            )
        elif isinstance(output, set):
            warnings.warn(
                "The 'predict' method returned a set instead of a list of predictions. "
                "When batching is enabled, 'predict' must return a list to handle multiple inputs correctly. "
                "Please update the 'predict' method to return a list of predictions to avoid unexpected behavior.",
                UserWarning,
            )
        return list(output)

    def _unbatch_stream(self, output_stream):
        for output in output_stream:
            yield list(output)

    def unbatch(self, output):
        """Convert a batched output to a list of outputs.

        When using batching, the predict method should return a list-like structure
        (list, tuple, or array) where each element corresponds to one input.

        For example, for a batch of 2 inputs, predict should return:
            [output1, output2]  # Correct

        Not:
            {"key1": [val1, val2], "key2": [val3, val4]}  # Incorrect

        If you need to return dictionaries, return a list of dicts:
            [{"key1": val1, "key2": val3}, {"key1": val2, "key2": val4}]  # Correct

        """
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

    def pre_setup(self, spec: Optional[LitSpec] = None):
        spec = spec or self._spec
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

    def health(self) -> Union[bool, Awaitable[bool]]:
        """Check the additional health status of the API.

        This method is used in the /health endpoint of the server to determine the health status.
        Users can extend this method to include additional health checks specific to their application.

        The default implementation is synchronous but users may optionally implement an
        ``async`` version which will be awaited by :class:`~litserve.server.LitServer`.

        Returns:
            bool: ``True`` if the API is healthy, ``False`` otherwise.

        """
        return True

    @property
    def loop(self):
        if self._loop == "auto":
            from litserve.loops.loops import get_default_loop

            self._loop = get_default_loop(self.stream, self.max_batch_size, self.enable_async)
        return self._loop

    @loop.setter
    def loop(self, value: "LitLoop"):
        self._loop = value

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value: LitSpec):
        self._spec = value

    @property
    def api_path(self):
        if self._spec:
            return self._spec.api_path
        return self._api_path

    @api_path.setter
    def api_path(self, value: str):
        self._api_path = value
