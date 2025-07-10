import numpy as np

import litserve as ls


class EmbeddingsAPI(ls.LitAPI):
    def setup(self, device):
        def model(x):
            return np.random.rand(len(x), 768)

        self.model = model

    def predict(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    api = EmbeddingsAPI(max_batch_size=10, batch_timeout=2, spec=ls.OpenAIEmbeddingSpec())
    server = ls.LitServer(api)
    server.run(port=8000)
