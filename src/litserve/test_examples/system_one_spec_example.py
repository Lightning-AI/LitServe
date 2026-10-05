import numpy as np

from litserve.api import LitAPI
from litserve.specs.system_one import DecisionOutput, DecisionRequest

# fixed logits for the questions in the `system_one_request_data` test fixture
TEST_LOGITS = {
    "department": {"returns": 0.0, "shipping": -1.0, "billing": 2.0},
    "escalate": [1.0, 0.0],
    "frustration": np.array([0.0, 1.0, 0.5]),
}


class TestDecisionAPI(LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, request: DecisionRequest) -> dict:
        return {q.id: TEST_LOGITS.get(q.id, [0.0] * len(q.options)) for q in request.questions}


class TestDecisionBatchedAPI(TestDecisionAPI):
    def predict(self, requests: list[DecisionRequest]) -> list[dict]:
        return [super(TestDecisionBatchedAPI, self).predict(request) for request in requests]


class TestDecisionAsyncAPI(TestDecisionAPI):
    async def predict(self, request: DecisionRequest) -> dict:
        return super().predict(request)


class TestDecisionAPIWithUsage(TestDecisionAPI):
    def predict(self, request: DecisionRequest) -> DecisionOutput:
        return DecisionOutput(logits=super().predict(request), input_tokens=42, output_tokens=3)


class TestDecisionAPIWithOutput(TestDecisionAPI):
    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)
        self.output = output

    def predict(self, request: DecisionRequest):
        return self.output


class TestDecisionAPIWithYieldPredict(TestDecisionAPI):
    def predict(self, request: DecisionRequest):
        yield {}


class TestDecisionAPIWithYieldEncodeResponse(TestDecisionAPI):
    def encode_response(self, output):
        yield output
