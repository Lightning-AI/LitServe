from abc import ABC, abstractmethod


class LitAPI(ABC):
    @abstractmethod
    def setup(self, devices):
        pass

    @abstractmethod
    def decode_request(self, request):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def encode_response(self, output):
        pass
