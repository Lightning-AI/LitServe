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


class AsyncAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x: x

    async def decode_request(self, request):
        return request["input"]

    async def predict(self, x):
        for i in range(10):
            yield self.model(i)

    async def encode_response(self, output):
        async for out in output:
            yield {"output": out}


if __name__ == "__main__":
    api = AsyncAPI(enable_async=True, stream=True)
    server = ls.LitServer(api)
    server.run(port=8000)
