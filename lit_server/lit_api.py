from abc import ABC, abstractmethod


class LitAPI(ABC):
    @abstractmethod
    def setup(self, devices):
        """
        Setup the model so it can be called in `predict`.
        """
        pass

    @abstractmethod
    def decode_request(self, request):
        """
        Convert the request payload to your model input.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Run the model on the input and return the output.
        """
        pass

    @abstractmethod
    def encode_response(self, output):
        """
        Convert the model output to a response payload.
        """
        pass
