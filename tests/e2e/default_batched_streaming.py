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

import litserve as ls


class SimpleStreamAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x, y: x * y

    def decode_request(self, request):
        return np.asarray(request["input"])

    def predict(self, x):
        for i in range(10):
            yield self.model(x, i)

    def encode_response(self, output_stream):
        for outputs in output_stream:
            yield [{"output": output} for output in outputs]


if __name__ == "__main__":
    server = ls.LitServer(SimpleStreamAPI(), stream=True, max_batch_size=4, batch_timeout=0.2, fast_queue=True)
    server.run(port=8000)
