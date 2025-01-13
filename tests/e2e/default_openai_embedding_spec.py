import litserve as ls
from litserve import OpenAIEmbeddingSpec
from litserve.test_examples.openai_embedding_spec_example import TestEmbedAPI

if __name__ == "__main__":
    server = ls.LitServer(TestEmbedAPI(), spec=OpenAIEmbeddingSpec(), fast_queue=True)
    server.run()
