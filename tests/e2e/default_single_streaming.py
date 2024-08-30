from litserve import LitServer, LitAPI


class SimpleStreamingAPI(LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x, y: x * y

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        for i in range(1, 4):
            yield self.model(i, x)

    def encode_response(self, output_stream):
        for output in output_stream:
            yield {"output": output}


if __name__ == "__main__":
    api = SimpleStreamingAPI()
    server = LitServer(api, stream=True)
    server.run(port=8000)
