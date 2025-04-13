from litserve import LitAPI


class WebSocketLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: f"Processed: {x}"

    def decode_request(self, request):
        print(f"Decoding request: {request}")
        return request.get("input", "default_input")

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}
