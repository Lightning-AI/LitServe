import json
import os
import requests
import subprocess
import time


def test_e2e_default_batching():
    process = subprocess.Popen(
        ["python", "tests/e2e/default_batching.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    time.sleep(5)
    assert os.path.exists("client.py"), f"Expected client file to be created at {os.getcwd()} after starting the server"
    output = subprocess.run("python client.py", shell=True, capture_output=True, text=True).stdout
    assert '{"output":16.0}' in output, "tests/simple_server.py didn't return expected output"
    os.remove("client.py")
    process.kill()


def test_e2e_batched_streaming():
    process = subprocess.Popen(
        ["python", "tests/e2e/default_batched_streaming.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    time.sleep(5)
    assert os.path.exists("client.py"), f"Expected client file to be created at {os.getcwd()} after starting the server"

    url = "http://127.0.0.1:8000/stream-predict"
    resp = requests.post(url, json={"input": 4.0}, headers=None, stream=True)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"

    outputs = []
    for line in resp.iter_content(chunk_size=4000):
        if line:
            outputs.append(json.loads(line.decode("utf-8")))

    assert len(outputs) == 10, "streaming server should have 10 outputs"
    assert {"output": 16.0} in outputs, "server didn't return expected output"
    process.kill()
