import litserve as ls
from litserve.test_examples.openai_spec_example import OpenAIBatchContext

if __name__ == "__main__":
    api = OpenAIBatchContext()
    server = ls.LitServer(api, spec=ls.OpenAISpec(), max_batch_size=2, batch_timeout=0.5, fast_queue=True)
    server.run(port=8000)
