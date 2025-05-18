from typing import List

import numpy as np

from litserve.api import LitAPI


class TestEmbedAPI(LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x) -> List[List[float]]:
        n = len(x) if isinstance(x, list) else 1
        return np.random.rand(n, 768).tolist()

    def encode_response(self, output) -> dict:
        return {"embeddings": output}


class TestEmbedBatchedAPI(TestEmbedAPI):
    def predict(self, batch) -> List[List[List[float]]]:
        return [np.random.rand(len(x), 768).tolist() for x in batch]


class TestEmbedAPIWithUsage(TestEmbedAPI):
    def encode_response(self, output) -> dict:
        return {"embeddings": output, "prompt_tokens": 10, "total_tokens": 10}


class TestEmbedAPIWithYieldPredict(TestEmbedAPI):
    def predict(self, x):
        yield from np.random.rand(768).tolist()


class TestEmbedAPIWithYieldEncodeResponse(TestEmbedAPI):
    def encode_response(self, output):
        yield {"embeddings": output}


class TestEmbedAPIWithNonDictOutput(TestEmbedAPI):
    def encode_response(self, output):
        return output


class TestEmbedAPIWithMissingEmbeddings(TestEmbedAPI):
    def encode_response(self, output):
        return {"output": output}
