# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

from fastapi import HTTPException
from litserve.specs.openai import ChatCompletionRequest
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


def test_decode_request():
    request = {"input": 4.0}
    api = ls.examples.SimpleLitAPI()
    assert api.decode_request(request) == 4.0, "Decode request should return the input 4.0"


def test_decode_request_with_openai_spec():
    api = ls.examples.TestAPI()
    api._sanitize(max_batch_size=1, spec=ls.OpenAISpec())
    request = ChatCompletionRequest(messages=[{"role": "system", "content": "Hello"}])
    decoded_request = api.decode_request(request)
    assert decoded_request[0]["content"] == "Hello", "Decode request should return the input message"


def test_decode_request_with_openai_spec_wrong_request():
    api = ls.examples.TestAPI()
    api._sanitize(max_batch_size=1, spec=ls.OpenAISpec())
    with pytest.raises(AttributeError, match="object has no attribute 'messages'"):
        api.decode_request({"input": "Hello"})


def test_encode_response():
    response = 4.0
    api = ls.examples.SimpleLitAPI()
    assert api.encode_response(response) == {"output": 4.0}, 'Encode response returns encoded output {"output": 4.0}'


def test_encode_response_with_openai_spec():
    api = ls.examples.TestAPI()
    api._sanitize(max_batch_size=1, spec=ls.OpenAISpec())
    response = "This is a LLM generated text".split()
    generated_tokens = []
    for output in api.encode_response(response):
        generated_tokens.append(output["content"])
    assert generated_tokens == response, f"Encode response should return the generated tokens {response}"


def test_encode_response_with_openai_spec_dict_token_usage():
    prompt = "This is a LLM generated text"

    def predict():
        for token in prompt.split():
            yield {"content": token, "prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}

    generated_tokens = []
    api = ls.examples.TestAPI()
    api._sanitize(max_batch_size=1, spec=ls.OpenAISpec())

    for output in api.encode_response(predict()):
        generated_tokens.append(output["content"])
    assert generated_tokens == prompt.split(), f"Encode response should return the generated tokens {prompt.split()}"


def test_encode_response_with_custom_spec_api():
    class CustomSpecAPI(ls.examples.TestAPI):
        def encode_response(self, output_stream):
            for output in output_stream:
                yield {"content": output}

    api = ls.examples.TestAPI()
    api._sanitize(max_batch_size=1, spec=CustomSpecAPI())
    response = "This is a LLM generated text".split()
    generated_tokens = []
    for output in api.encode_response(response):
        generated_tokens.append(output["content"])
    assert generated_tokens == response, f"Encode response should return the generated tokens {response}"


def test_encode_response_with_openai_spec_invalid_input():
    api = ls.examples.TestAPI()
    api._sanitize(max_batch_size=1, spec=ls.OpenAISpec())
    response = 10
    with pytest.raises(TypeError, match="object is not iterable"):
        next(api.encode_response(response))


def test_encode_response_with_openai_spec_invalid_predict_output():
    def predict():
        yield {"hello": "world"}

    api = ls.examples.TestAPI()
    api._sanitize(max_batch_size=1, spec=ls.OpenAISpec())
    with pytest.raises(HTTPException, match=r"500: Malformed output from LitAPI.predict"):
        next(api.encode_response(predict()))
