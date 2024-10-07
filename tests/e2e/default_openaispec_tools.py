import litserve as ls
from litserve import OpenAISpec
from litserve.specs.openai import ChatMessage
from litserve.test_examples.openai_spec_example import TestAPI


class TestAPIWithToolCalls(TestAPI):
    def encode_response(self, output):
        yield ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_current_weather", "arguments": '{\n"location": "Boston, MA"\n}'},
                }
            ],
        )


if __name__ == "__main__":
    server = ls.LitServer(TestAPIWithToolCalls(), spec=OpenAISpec())
    server.run()
