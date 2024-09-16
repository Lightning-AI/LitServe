import litserve as ls
from fastapi.testclient import TestClient
from litserve.utils import wrap_litserve_start


def test_callback():
    lit_api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(lit_api)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.json() == {"output": 16.0}
    #
    #
    #     # Read the log file
    #     with open(log_file.name, 'r') as f:
    #         log_contents = f.read()
    #
    # assert "Prediction took" in log_contents
