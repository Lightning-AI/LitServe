import litserve as ls
from fastapi.testclient import TestClient
import logging
import tempfile


from litserve.utils import wrap_litserve_start

# Create a temporary file for logging
log_file = tempfile.NamedTemporaryFile(delete=False)
log_file.close()
# Ensure logging is configured to output to the temporary file
logging.basicConfig(filename=log_file.name, level=logging.INFO)


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
