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
from typing import List, Dict, Union

import litserve as ls
from litserve.specs.openai import OpenAISpec
import logging

logging.basicConfig(level=logging.DEBUG)


class OpenAILitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def predict(self, x: Dict[str, str]) -> Union[List[Dict[str, str]], Dict[str, str]]:
        yield {"role": "assistant", "content": "This is a generated output"}


class OpenAILitAPIWithCustomEncode(ls.LitAPI):
    def setup(self, device):
        self.model = None

    def encode_response(self, output_generator):
        for output in output_generator:
            output["content"] = "This output is a custom encoded output"
            yield output

    def predict(self, x: Dict[str, str]) -> Union[List[Dict[str, str]], Dict[str, str]]:
        yield {"role": "assistant", "content": "This is a generated output"}


if __name__ == "__main__":
    server = ls.LitServer(OpenAILitAPIWithCustomEncode(), spec=OpenAISpec(), stream=True)
    server.run(port=8000)
