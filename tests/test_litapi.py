import numpy as np

import litserve as ls


class TestDefaultBatchedAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x: len(x)

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


class TestCustomBatchedAPI(TestDefaultBatchedAPI):
    def batch(self, inputs):
        return np.stack(inputs)

    def unbatch(self, output):
        return list(output)


class TestStreamAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = None

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        # x is a list of integers
        for i in range(4):
            yield np.asarray(x) * i

    def encode_response(self, output_stream):
        for output in output_stream:
            output = list(output)
            yield [{"output": o} for o in output]


def test_default_batch_unbatch():
    api = TestDefaultBatchedAPI()
    api._sanitize(max_batch_size=4, spec=None)
    inputs = [1, 2, 3, 4]
    output = api.batch(inputs)
    assert output == inputs
    assert api.unbatch(output) == inputs


def test_custom_batch_unbatch():
    api = TestCustomBatchedAPI()
    api._sanitize(max_batch_size=4, spec=None)
    inputs = [1, 2, 3, 4]
    output = api.batch(inputs)
    assert np.all(output == np.array(inputs))
    assert api.unbatch(output) == inputs


def test_batch_unbatch_stream():
    api = TestStreamAPI()
    api._sanitize(max_batch_size=4, spec=None)
    inputs = [1, 2, 3, 4]
    output = api.batch(inputs)
    assert np.all(output == np.array(inputs))

    pass
