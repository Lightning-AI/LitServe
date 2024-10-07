from litserve.api import LitAPI


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        # Set up the model, so it can be called in `predict`.
        self.model = lambda x: x**2

    def decode_request(self, request):
        # Convert the request payload to your model input.
        return request["input"]

    def predict(self, x):
        # Run the model on the input and return the output.
        return self.model(x)

    def encode_response(self, output):
        # Convert the model output to a response payload.
        return {"output": output}


class SimpleBatchedAPI(LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x: x**2

    def decode_request(self, request):
        import numpy as np

        return np.asarray(request["input"])

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


class SimpleTorchAPI(LitAPI):
    def setup(self, device):
        # move the model to the correct device
        # keep track of the device for moving data accordingly
        import torch.nn as nn

        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.linear.weight.data.fill_(2.0)
                self.linear.bias.data.fill_(1.0)

            def forward(self, x):
                return self.linear(x)

        self.model = Linear().to(device)

    def decode_request(self, request):
        import torch

        # get the input and create a 1D tensor on the correct device
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def predict(self, x):
        # the model expects a batch dimension, so create it
        return self.model(x[None, :])

    def encode_response(self, output):
        # float will take the output value directly onto CPU memory
        return {"output": float(output)}


class SimpleStreamAPI(LitAPI):
    """
    Run as:
        ```
        server = ls.LitServer(SimpleStreamAPI(), stream=True)
        server.run(port=8000)
        ```
    Then, in a new Python session, retrieve the responses as follows:
        ```
        import requests
        url = "http://127.0.0.1:8000/predict"
        resp = requests.post(url, json={"input": "Hello world"}, headers=None, stream=True)
        for line in resp.iter_content(5000):
        if line:
            print(line.decode("utf-8"))
        ```
    """

    def setup(self, device) -> None:
        self.model = lambda x, y: f"{x}: {y}"

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        for i in range(3):
            yield self.model(i, x)

    def encode_response(self, output_stream):
        for output in output_stream:
            yield {"output": output}
