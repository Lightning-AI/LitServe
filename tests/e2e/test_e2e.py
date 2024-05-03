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
import json
import requests
import subprocess
import time


def test_e2e_default_api():
    process = subprocess.Popen(
        ["python", "tests/e2e/default_api.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    time.sleep(5)
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"
    assert resp.json() == {"output": 16.0}, "tests/simple_server.py didn't return expected output"
    process.kill()


def test_e2e_default_batching():
    process = subprocess.Popen(
        ["python", "tests/e2e/default_batching.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    time.sleep(5)
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"
    assert resp.json() == {"output": 16.0}, "tests/simple_server.py didn't return expected output"
    process.kill()


def test_e2e_batched_streaming():
    process = subprocess.Popen(
        ["python", "tests/e2e/default_batched_streaming.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    time.sleep(5)
    resp = requests.post("http://127.0.0.1:8000/stream-predict", json={"input": 4.0}, headers=None, stream=True)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"

    outputs = []
    for line in resp.iter_content(chunk_size=4000):
        if line:
            outputs.append(json.loads(line.decode("utf-8")))

    assert len(outputs) == 10, "streaming server should have 10 outputs"
    assert {"output": 16.0} in outputs, "server didn't return expected output"
    process.kill()
