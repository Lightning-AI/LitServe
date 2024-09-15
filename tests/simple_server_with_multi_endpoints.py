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
from litserve.server import LitServer, run_all
from litserve.test_examples import SimpleLitAPI


class SimpleLitAPI1(SimpleLitAPI):
    def setup(self, device):
        self.model = lambda x: x**1


class SimpleLitAPI2(SimpleLitAPI):
    def setup(self, device):
        self.model = lambda x: x**2


class SimpleLitAPI3(SimpleLitAPI):
    def setup(self, device):
        self.model = lambda x: x**3


class SimpleLitAPI4(SimpleLitAPI):
    def setup(self, device):
        self.model = lambda x: x**4


if __name__ == "__main__":
    server1 = LitServer(SimpleLitAPI1(), accelerator="cpu", devices=1, timeout=10, api_path="/predict-1")
    server2 = LitServer(SimpleLitAPI2(), accelerator="cpu", devices=1, timeout=10, api_path="/predict-2")
    server3 = LitServer(SimpleLitAPI3(), accelerator="cpu", devices=1, timeout=10, api_path="/predict-3")
    server4 = LitServer(SimpleLitAPI4(), accelerator="cpu", devices=1, timeout=10, api_path="/predict-4")
    run_all([server1, server2, server3, server4], port=8000)
