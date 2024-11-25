from typing import List

import numpy as np

from litserve.api import LitAPI


class TestAPI(LitAPI):
    def setup(self, device):
        self.model = None

    def decode_request(self, request) -> List[str]:
        return request.get_input_as_list()

    def predict(self, x) -> List[List[float]]:
        return np.random.rand(len(x), 768).tolist()

    def encode_response(self, output) -> dict:
        return {"embeddings": output}
