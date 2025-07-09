import multiprocessing as mp
import time

from fastapi import Request, Response

from litserve import LitAPI, LitServer


class CrashingLitAPI(LitAPI):
    """API that crashes after 2 predictions to simulate worker failure."""

    def setup(self, device):
        self.model = lambda x: x**2
        self.count = 0

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_worker_monitoring_triggers_shutdown_on_worker_death():
    """Test that server shuts down when a worker dies."""
    server = LitServer(CrashingLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    # Track if shutdown was called
    shutdown_called = {"value": False}

    def mock_shutdown(manager, uvicorn_workers, shutdown_reason="normal"):
        shutdown_called["value"] = True

    manager = mp.Manager()
    server._shutdown_event = manager.Event()
    server._perform_graceful_shutdown = mock_shutdown
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=SystemExit,
        args=("Crash",),
        name="crashed process",
    )

    server.inference_workers = [proc]

    server._start_worker_monitoring(manager, [])

    server._start_worker_monitoring(manager, [])

    for i in range(20):
        if shutdown_called["value"]:
            break
        assert i != 19, "Server should shutdown when worker dies"
        time.sleep(1)


if __name__ == "__main__":
    test_worker_monitoring_triggers_shutdown_on_worker_death()
    print("Test passed!")
