from pydantic import BaseModel
from transformers import pipeline

import litserve as ls
from litserve.mcp import MCP


class TextClassificationRequest(BaseModel):
    input: str


class TextClassificationAPI(ls.LitAPI):
    def setup(self, device):
        self.model = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model", device=device)

    def decode_request(self, request: TextClassificationRequest):
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return output[0]


if __name__ == "__main__":
    api = TextClassificationAPI(mcp=MCP(description="Classifies sentiment in text"))
    server = ls.LitServer(api)
    server.run(port=8000)
