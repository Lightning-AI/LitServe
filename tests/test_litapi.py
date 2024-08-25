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
    assert output == inputs, "Default batch should not change input"
    assert api.unbatch(output) == inputs, "Default unbatch should not change input"


def test_custom_batch_unbatch():
    api = TestCustomBatchedAPI()
    api._sanitize(max_batch_size=4, spec=None)
    inputs = [1, 2, 3, 4]
    output = api.batch(inputs)
    assert np.all(output == np.array(inputs)), "Custom batch stacks input as numpy array"
    assert api.unbatch(output) == inputs, "Custom unbatch should unstack input as list"


def test_batch_unbatch_stream():
    api = TestStreamAPI()
    api._sanitize(max_batch_size=4, spec=None)
    inputs = [1, 2, 3, 4]
    output = api.batch(inputs)
    output = api.predict(output)
    output = api.unbatch(output)
    output = api.encode_response(output)
    first_resp = [o["output"] for o in next(output)]
    expected_outputs = [[0, 0, 0, 0], [1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]]
    assert first_resp == expected_outputs[0], "First response should be 0s"
    count = 1
    for out, expected_output in zip(output, expected_outputs[1:]):
        resp = [o["output"] for o in out]
        assert resp == expected_output
        count += 1

    assert count == 4, "Should have 4 responses"
