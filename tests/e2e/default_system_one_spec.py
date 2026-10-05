import litserve as ls
from litserve import SystemOneSpec
from litserve.test_examples.system_one_spec_example import TestDecisionAPI

if __name__ == "__main__":
    api = TestDecisionAPI(spec=SystemOneSpec())
    server = ls.LitServer(api, fast_queue=True)
    server.run()
