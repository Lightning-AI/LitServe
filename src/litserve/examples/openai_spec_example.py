import litserve as ls
from litserve.specs.openai import OpenAISpec


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = ...

    def decode_request(self, request):
        return request

    def predict(self, x):
        return "This is a generated output"

    def encode_response(self, output):
        return {"text": output}


if __name__ == "__main__":
    spec = OpenAISpec()
    server = ls.LitServer(SimpleLitAPI(), spec=spec)
    server.run(port=8000)
