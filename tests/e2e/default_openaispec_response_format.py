import litserve as ls
from litserve import OpenAISpec
from litserve.test_examples.openai_spec_example import TestAPIWithStructuredOutput

if __name__ == "__main__":
    server = ls.LitServer(TestAPIWithStructuredOutput(), spec=OpenAISpec())
    server.run()
