import pytest
from fastapi.testclient import TestClient

import litserve as ls
from litserve.utils import wrap_litserve_start


def test_info_route():
    model_metadata = {"name": "my-awesome-model", "version": "v1.1.0"}
    expected_response = {
        "model": {
            "name": "my-awesome-model",
            "version": "v1.1.0",
        },
        "server": {
            "devices": ["cpu"],
            "workers_per_device": 1,
            "timeout": 30,
            "max_batch_size": 1,
            "batch_timeout": 0.0,
            "stream": False,
            "max_payload_size": None,
            "track_requests": False,
        },
    }

    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), accelerator="cpu", model_metadata=model_metadata)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.get("/info", headers={"Host": "localhost"})
        assert response.status_code == 200, f"Expected response to be 200 but got {response.status_code}"
        assert response.json() == expected_response, "server didn't return expected output"


def test_model_metadata_json_error():
    with pytest.raises(ValueError, match="model_metadata is not JSON serializable"):
        ls.LitServer(ls.test_examples.SimpleLitAPI(), model_metadata=int)
