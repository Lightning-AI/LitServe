from litserve.server import LitServer
from litserve.api import LitAPI
from fastapi import Request, Response


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


if __name__ == "__main__":
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, timeout=10)
    server.run()
