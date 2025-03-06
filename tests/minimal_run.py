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
import subprocess
import time
import urllib.request

import psutil


def main():
    process = subprocess.Popen(
        ["lightning", "serve", "api", "tests/simple_server.py"],
    )
    print("Waiting for server to start...")
    time.sleep(10)
    try:
        url = "http://127.0.0.1:8000/predict"
        data = json.dumps({"input": 4.0}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        response = urllib.request.urlopen(request)
        status_code = response.getcode()
        assert status_code == 200
    except Exception:
        raise

    finally:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        process.kill()


if __name__ == "__main__":
    main()
