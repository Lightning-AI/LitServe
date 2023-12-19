from lit_server import LitAPI, LitServer


# vanilla

class SimpleLitAPI(LitAPI):
    def setup(self, devices):
        self.model = lambda x: x**2

    def predict(self, x):
        return self.model(x)

    def decode_request(self, request):
        return request["input"]

    def encode_response(self, output):
        return {"output": output}


# with pydantic (advantage: FastAPI does schema validation)

from pydantic import BaseModel


class PredictRequest(BaseModel):
    input: float


class PredictResponse(BaseModel):
    output: float


class SimpleLitAPI2(LitAPI):
    def setup(self, devices):
        self.model = lambda x: x**2

    def predict(self, x):
        return self.model(x)

    def decode_request(self, request: PredictRequest) -> float:
        return request.input

    def encode_response(self, output: float) -> PredictResponse:
        return PredictResponse(output=output)


if __name__ == "__main__":
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1)
    server.run(port=8888)
