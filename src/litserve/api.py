# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from abc import ABC, abstractmethod


class LitAPI(ABC):
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
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """Run the model on the input and return the output."""
        pass

    def unbatch(self, inputs):
        """Convert a batched output to a list of outputs."""
        raise NotImplementedError

    @abstractmethod
    def encode_response(self, output):
        """Convert the model output to a response payload."""
        pass
