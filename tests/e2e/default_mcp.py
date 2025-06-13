from pydantic import BaseModel

import litserve as ls
from litserve.mcp import MCP


class PowerRequest(BaseModel):
    input: float


class MyLitAPI(ls.test_examples.SimpleLitAPI):
    def decode_request(self, request: PowerRequest) -> int:
        print(f"Decoding request: {request}")
        return request.input


if __name__ == "__main__":
    api = MyLitAPI(mcp=MCP(description="Returns the power of a number."))
    server = ls.LitServer(api)
    server.run(port=8000)
