# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import ABC, abstractmethod


def no_batch_unbatch_message(obj, data):
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
        # consider assigning an implementation when starting server
        # to avoid the runtime cost of checking (should be negligible)
        if hasattr(inputs[0], "__torch_function__"):
            import torch

            return torch.stack(inputs)
        if inputs[0].__class__.__name__ == "ndarray":
            import numpy

            return numpy.stack(inputs)
        raise NotImplementedError(no_batch_unbatch_message(self, inputs))

    @abstractmethod
    def predict(self, x):
        """Run the model on the input and return the output."""
        pass

    def unbatch(self, output):
        """Convert a batched output to a list of outputs."""
        if hasattr(output, "__torch_function__") or output.__class__.__name__ == "ndarray":
            return list(output)
        raise NotImplementedError(no_batch_unbatch_message(self, output))

    @abstractmethod
    def encode_response(self, output):
        """Convert the model output to a response payload."""
        pass
