import numpy as np
import litserve as ls


class SimpleStreamAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x: x**2

    def decode_request(self, request):
        return np.asarray(request["input"])

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = SimpleStreamAPI()
    server = ls.LitServer(api, max_batch_size=4, batch_timeout=0.05)
    server.run(port=8000)
