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
    specs = OpenAISpec()
    server = ls.LitServer(SimpleLitAPI(), specs=specs)
    server.run(port=8000)
