# lit-server

Lightweight inference server for AI/ML models based on FastAPI.

lit-server supports:

- serving models across multiple GPUs
- full flexibility in the definition of request and response
- automatic schema validation
- handling of timeouts and disconnections
- API key authentication

While being fairly capable and fast, the server is extremely simple and hackable.

For example, adding batching and streaming logic should be relatively straightforward
and we'll probably add those directly as part of the library in the near future.

## Install

```bash
pip install git+https://github.com/Lightning-AI/lit-server.git
```

## Use

### Creating a simple API server

```python
from lib import LitAPI

class SimpleLitAPI(LitAPI):
    def setup(self, device):
        """
        Setup the model so it can be called in `predict`.
        """
        self.model = lambda x: x**2

    def decode_request(self, request):
        """
        Convert the request payload to your model input.
        """
        return request["input"]

    def predict(self, x):
        """
        Run the model on the input and return the output.
        """
        return self.model(x)

    def encode_response(self, output):
        """
        Convert the model output to a response payload.
        """
        return {"output": output}
```

Now instantiate the API and start the server on an accelerator:

```python
from lib import LitServer

api = SimpleLitAPI()

server = LitServer(api, accelerator="cpu")
server.run(port=8000)
```

The server expects the client to send a `POST` to the `/predict` URL with a JSON payload.
The way the payload is structured is up to the implementation of the `LitAPI` subclass.

Once the server starts it generates an example client you can use like this:

```bash
python client.py
```

that simply posts a sample request to the server:

```python
response = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0})
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
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: PredictRequest) -> float:
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output: float) -> PredictResponse:
        return PredictResponse(output=output)


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cpu")
    server.run(port=8888)
```


### Serving on GPU

`LitServer` has the ability to coordinate serving from multiple GPUs.

For example, running the API server on a 4-GPU machine, with a PyTorch model served by each GPU:

```python
from fastapi import Request, Response

from lit_server import LitAPI, LitServer

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.fill_(2.0)
        self.linear.bias.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        # move the model to the correct device
        # keep track of the device for moving data accordingly
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request):
        # get the input and create a 1D tensor on the correct device
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def predict(self, x):
        # the model expects a batch dimension, so create it
        return self.model(x[None, :])

    def encode_response(self, output) -> Response:
        # float will take the output value directly onto CPU memory
        return {"output": float(output)}


if __name__ == "__main__":
    # accelerator="cuda", devices=4 will lead to 4 workers serving the
    # model from "cuda:0", "cuda:1", "cuda:2", "cuda:3" respectively
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=4)
    server.run(port=8000)
```

The `devices` variable can also be an array specifying what device id to
run the model on:

```python
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=[0, 3])
```

Last, you can run multiple copies of the same model from the same device,
if the model is small. The following will load two copies of the model on
each of the 4 GPUs:

```python
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=4, workers_per_device=2)
```

### Timeouts and disconnections

The server will remove a queued request if the client requesting it disconnects.

You can configure a timeout (in seconds) after which clients will receive a `504` HTTP
response (Gateway Timeout) indicating that their request has timed out.

For example, this is how you can configure the server with a timeout of 30 seconds per response.

```python
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=4, timeout=30)
```

This is useful to avoid requests queuing up beyond the ability of the server to respond.

### Using API key authentication

In order to secure the API behind an API key, just define the env var when 
starting the server

```bash
LIT_SERVER_API_KEY=supersecretkey python main.py
```

Clients are expected to auth with the same API key set in the `X-API-Key` HTTP header.

## License

lit-server is released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
See LICENSE file for details.
