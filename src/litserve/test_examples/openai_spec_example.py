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
import time

from litserve.api import LitAPI
from litserve.specs.openai import ChatMessage


class TestAPI(LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        yield "This is a generated output"


class TestAPIWithCustomEncode(TestAPI):
    def encode_response(self, output):
        yield ChatMessage(role="assistant", content="This is a custom encoded output")


class TestAPIWithToolCalls(TestAPI):
    def encode_response(self, output):
        yield ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "function_1", "arguments": '{"arg_1": "arg_1_value"}'},
                }
            ],
        )


class TestAPIWithStructuredOutput(TestAPI):
    def encode_response(self, output):
        yield ChatMessage(
            role="assistant",
            content='{"name": "Science Fair", "date": "Friday", "participants": ["Alice", "Bob"]}',
        )


class OpenAIBatchContext(LitAPI):
    def setup(self, device: str) -> None:
        self.model = None

    def batch(self, inputs):
        return inputs

    def predict(self, inputs, context):
        n = len(inputs)
        assert isinstance(context, list)
        for ctx in context:
            ctx["temperature"] = 1.0
        output = (
            "Hi! It's nice to meet you. Is there something I can help you with or would you like to chat?"
        ).split()
        for out in output:
            time.sleep(0.01)  # fake delay
            yield [out + " "] * n

    def unbatch(self, output):
        return output

    def encode_response(self, output_stream, context):
        for outputs in output_stream:
            yield [{"role": "assistant", "content": output} for output in outputs]
        for ctx in context:
            assert ctx["temperature"] == 1.0, f"context {ctx} is not 1.0"


class OpenAIWithUsage(LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        yield {
            "role": "assistant",
            "content": "10 + 6 is equal to 16.",
            "prompt_tokens": 25,
            "completion_tokens": 10,
            "total_tokens": 35,
        }


class OpenAIWithUsageEncodeResponse(LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        # streaming tokens
        yield from ["10", " +", " ", "6", " is", " equal", " to", " ", "16", "."]

    def encode_response(self, output):
        for out in output:
            yield {"role": "assistant", "content": out}

        yield {"role": "assistant", "content": "", "prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35}


class OpenAIBatchingWithUsage(OpenAIWithUsage):
    def batch(self, inputs):
        return inputs

    def predict(self, x):
        n = len(x)
        yield ["10 + 6 is equal to 16."] * n

    def encode_response(self, output_stream_batch, context):
        n = len(context)
        for output_batch in output_stream_batch:
            yield [{"role": "assistant", "content": out} for out in output_batch]

        yield [
            {"role": "assistant", "content": "", "prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35}
        ] * n

    def unbatch(self, output):
        return output
