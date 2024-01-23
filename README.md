# lit-server

Lightweight inference server for AI/ML models based on FastAPI.

Lit-server supports:

- serving models on multiple devices (multi-GPU)
- full flexibility in the definition of request and response
- handling of timeouts and disconnection
- API key authentication
- automatic schema validation

While being fairly capable and fast, the server is extremely simple and hackable.

## Install

```bash
pip install git+https://github.com/Lightning-AI/lit-server.git
```

## Use

### Creating a simple API server

```python
from lib import LitAPI

class SimpleLitAPI(LitAPI):
    def setup(self, devices):
        self.model = lambda x: x**2

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}
```

Now instantiate the API and start the server on an accelerator:

```python
from lib import LitServer

api = SimpleLitAPI()

server = LitServer(api, accelerator="cuda", devices=[0, 1])
server.run(port=8888)
```

Once the server starts it generates an example client you can use like this:

```bash
python client.py
```

### Pydantic models

You can also define request and response as [Pydantic models](https://docs.pydantic.dev/latest/),
with the advantage that the API will automatically validate the request.

```python
from pydantic import BaseModel

class PredictRequest(BaseModel):
    input: float

class PredictResponse(BaseModel):
    output: float

class SimpleLitAPI2(LitAPI):
    def setup(self, devices):
        self.model = lambda x: x**2

    def decode_request(self, request: PredictRequest) -> float:
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output: float) -> PredictResponse:
        return PredictResponse(output=output)


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cpu", devices=1)
    server.run(port=8888)
```