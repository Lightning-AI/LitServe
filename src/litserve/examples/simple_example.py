import litserve as ls


class SimpleLitAPI(ls.LitAPI):
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


class SimpleBatchedAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x: x**2

    def decode_request(self, request):
        import numpy as np

        return np.asarray(request["input"])

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


class SimplePyTorchAPI(ls.LitAPI):
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
        self.device = device

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
