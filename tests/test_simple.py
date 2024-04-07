# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
import socket
import time

from fastapi import Request, Response
from fastapi.testclient import TestClient

from litserve import LitAPI, LitServer

import requests


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_simple():
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=5)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}


def make_load_request(server, outputs):
    with TestClient(server.app) as client:
        for _ in range(100):
            response = client.post("/predict", json={"input": 4.0})
            outputs.append(response.json())


def test_load():
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=25)

    from threading import Thread

    threads = []

    for _ in range(1):
        outputs = []
        t = Thread(target=make_load_request, args=(server, outputs))
        t.start()
        threads.append((t, outputs))

    for t, outputs in threads:
        t.join()
        for el in outputs:
            assert el == {"output": 16.0}


class SlowLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        import time

        time.sleep(1)
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_timeout():
    server = LitServer(SlowLitAPI(), accelerator="cpu", devices=1, timeout=0.5)

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 504


def get_free_port(port=1024, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise OSError("no free ports")


def start_server_slow(port):
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)
    server.run(port=port)


def make_request(port, res_queue):
    response = requests.post(f"http://127.0.0.1:{port}/predict", json={"input": 4.0})
    res_queue.put(response.json())


def test_concurrent_requests():
    n_requests = 100

    port = get_free_port()

    p = Process(target=start_server_slow, args=(port,))
    p.start()

    time.sleep(1)

    res_queue = Queue()
    futures = []
    with ThreadPoolExecutor(max_workers=n_requests) as executor:
        for _ in range(n_requests):
            futures.append(executor.submit(make_request, port, res_queue))

    time.sleep(0.01)
    p.kill()

    count = 0
    while True:
        try:
            response = res_queue.get_nowait()
            count += 1
        except Empty:
            break

        assert response == {"output": 16.0}

    assert count == n_requests
