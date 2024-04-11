from litserve.server import LitServer
import pytest
from litserve.api import LitAPI
from fastapi import Request, Response
from fastapi.testclient import TestClient


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


class SimpleStreamAPI(LitAPI):
    def setup(self, device):
        self.sentence = "LitServe is streaming output"

    def decode_request(self, request: Request) -> float:
        return request["prompt"]

    def predict(self, x):
        output = f"prompt={x} generated_output={self.sentence}".split()
        yield from output

    def encode_response(self, output: str) -> str:
        return output.lower()


@pytest.fixture()
def simple_litapi():
    return SimpleLitAPI()


@pytest.fixture()
def simple_stream_api():
    return SimpleStreamAPI()


@pytest.fixture()
def lit_server(simple_litapi):
    return LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)


@pytest.fixture()
def sync_testclient(lit_server):
    with TestClient(lit_server.app) as client:
        yield client
