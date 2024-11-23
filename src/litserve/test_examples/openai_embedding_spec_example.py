import numpy as np

from litserve.api import LitAPI


class TestAPI(LitAPI):
    def setup(self, device):
        self.model = None

    def decode_request(self, request):
        return request

    def predict(self, x):
        return [np.random.rand(768).tolist()]

    def encode_response(self, response):
        return {"embeddings": response}
