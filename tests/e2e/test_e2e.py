import json
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
