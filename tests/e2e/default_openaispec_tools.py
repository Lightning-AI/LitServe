import litserve as ls
from litserve import OpenAISpec
from litserve.examples.openai_spec_example import TestAPIWithToolCalls

if __name__ == "__main__":
    server = ls.LitServer(TestAPIWithToolCalls(), spec=OpenAISpec())
    server.run()
