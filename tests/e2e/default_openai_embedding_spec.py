import litserve as ls
from litserve import OpenAIEmbeddingSpec
from litserve.test_examples.openai_embedding_spec_example import TestAPI

if __name__ == "__main__":
    server = ls.LitServer(TestAPI(), spec=OpenAIEmbeddingSpec())
    server.run()
