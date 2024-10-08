import litserve as ls
from litserve import OpenAISpec
from litserve.test_examples.openai_spec_example import TestAPI

if __name__ == "__main__":
    server = ls.LitServer(TestAPI(), spec=OpenAISpec())
    server.run()
