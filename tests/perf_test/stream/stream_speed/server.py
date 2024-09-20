import litserve as ls


class SimpleStreamingAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = None

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        yield from range(10000)

    def encode_response(self, output_stream):
        for output in output_stream:
            yield {"output": output}


if __name__ == "__main__":
    api = SimpleStreamingAPI()
    server = ls.LitServer(
        api,
        stream=True,
    )
    server.run(port=8000, generate_client_file=False)
