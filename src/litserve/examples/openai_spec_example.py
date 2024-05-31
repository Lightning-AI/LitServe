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

import litserve as ls
from litserve.specs.openai import OpenAISpec, ChatMessage
import logging

logging.basicConfig(level=logging.DEBUG)


class TestAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x):
        yield "This is a generated output"


class TestAPIWithCustomEncode(TestAPI):
    def encode_response(self, output):
        yield ChatMessage(role="assistant", content="This is a custom encoded output")


class TestAPIWithToolsCalls(TestAPI):
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


if __name__ == "__main__":
    server = ls.LitServer(TestAPIWithCustomEncode(), spec=OpenAISpec())
    server.run(port=8000)
