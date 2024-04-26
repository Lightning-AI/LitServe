import numpy as np

import json
import litserve as ls


class SimpleStreamAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x, y: x * y

    def decode_request(self, request):
        return np.asarray(request["input"])

    def predict(self, x):
        for i in range(10):
            yield self.model(x, i)

    def encode_response(self, output_stream):
        for outputs in output_stream:
            yield [json.dumps({"output": output}) for output in outputs]


if __name__ == "__main__":
    server = ls.LitServer(SimpleStreamAPI(), stream=True, max_batch_size=4, batch_timeout=0.2)
    server.run(port=8000)
