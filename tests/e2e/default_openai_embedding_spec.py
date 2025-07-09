import litserve as ls
from litserve import OpenAIEmbeddingSpec
from litserve.test_examples.openai_embedding_spec_example import TestEmbedAPI

if __name__ == "__main__":
    api = TestEmbedAPI(spec=OpenAIEmbeddingSpec())
    server = ls.LitServer(api, fast_queue=True)
    server.run()
