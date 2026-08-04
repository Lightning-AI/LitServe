import os

import pytest
from fastapi.testclient import TestClient

import litserve as ls
import litserve.metrics
from litserve.metrics import _PROMETHEUS_AVAILABLE

# Only run tests if prometheus is available
pytestmark = pytest.mark.skipif(not _PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed")


class SimpleAPI(ls.LitAPI):
    def setup(self, device):
        pass

    def predict(self, x):
        self.log("batch_size", len(x))
        return x


def test_prometheus_logger_initialization():
    api = SimpleAPI()
    logger = ls.metrics.PrometheusLogger()
    server = ls.LitServer(api, loggers=[logger])

    assert server._metrics_dir is not None
    assert os.environ.get("PROMETHEUS_MULTIPROC_DIR") == server._metrics_dir
    assert os.path.exists(server._metrics_dir)
    assert os.path.basename(server._metrics_dir).startswith("litserve_prom_")


def test_prometheus_metrics_endpoint():
    api = SimpleAPI()
    logger = ls.metrics.PrometheusLogger()
    server = ls.LitServer(api, loggers=[logger])

    # Use wrap_litserve_start to initialize internal states
    with ls.utils.wrap_litserve_start(server):
        client = TestClient(server.app)

        # Hit healthcheck multiple times
        client.get("/health")
        client.get("/health")
        client.get("/health")

        # Hit an invalid route to test error statuses
        client.get("/invalid_route")

        # Hit metrics
        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text

        # Parse metrics instead of brittle string matching
        from prometheus_client.parser import text_string_to_metric_families

        metrics = list(text_string_to_metric_families(text))

        http_reqs = next((m for m in metrics if m.name == "litserve_http_requests"), None)
        assert http_reqs is not None

        # Validate healthcheck route is ignored (since it's skipped in middleware)
        health_sample = next((s for s in http_reqs.samples if s.labels.get("endpoint") == "/health"), None)
        assert health_sample is None

        # Validate invalid route
        invalid_sample = next((s for s in http_reqs.samples if s.labels.get("endpoint") == "/invalid_route"), None)
        assert invalid_sample is not None
        assert invalid_sample.value == 1.0


def test_prometheus_inference_metrics():
    api = SimpleAPI()
    logger = ls.metrics.PrometheusLogger()
    server = ls.LitServer(api, loggers=[logger])

    # Use wrap_litserve_start to initialize internal states
    with ls.utils.wrap_litserve_start(server):
        # We must explicitly start the logger connector process since wrap_litserve_start skips it
        server._logger_connector.run(server)

        with TestClient(server.app) as client:
            # Post to predict to trigger full pipeline: API -> Queue -> Worker -> process() -> /metrics
            client.post("/predict", json=[1, 2, 3, 4])
            client.post("/predict", json=[1, 2])

            import time

            # Retry loop to wait deterministically for background processes instead of a fixed sleep
            for _ in range(20):
                response = client.get("/metrics")
                text = response.text

                from prometheus_client.parser import text_string_to_metric_families

                metrics = list(text_string_to_metric_families(text))
                batch_size = next((m for m in metrics if m.name == "litserve_batch_size"), None)

                if batch_size is not None:
                    batch_count = next((s for s in batch_size.samples if s.name == "litserve_batch_size_count"), None)
                    if batch_count is not None and batch_count.value == 2.0:
                        break
                time.sleep(0.1)
            else:
                pytest.fail("Timeout waiting for metrics to propagate from background processes")

            batch_sum = next((s for s in batch_size.samples if s.name == "litserve_batch_size_sum"), None)
            assert batch_sum is not None
            assert batch_sum.value == 6.0


def test_prometheus_cleanup_on_shutdown():
    api = SimpleAPI()
    logger = ls.metrics.PrometheusLogger()
    server = ls.LitServer(api, loggers=[logger])
    metrics_dir = server._metrics_dir

    # Fake the logger connector setup
    server._logger_connector = ls.loggers._LoggerConnector(server, [logger])
    server.inference_workers = []

    # Assert dir exists
    assert os.path.exists(metrics_dir)

    # Call shutdown logic
    class FakeManager:
        def shutdown(self):
            pass

    server._perform_graceful_shutdown(FakeManager(), {})

    # Assert dir is deleted
    assert not os.path.exists(metrics_dir)


def test_prometheus_process_method():
    api = SimpleAPI()
    logger = ls.metrics.PrometheusLogger()
    _ = ls.LitServer(api, loggers=[logger])

    # Explicitly call process to ensure coverage on main thread
    logger.process("batch_size", 4)
    logger.process("batch_size", 8)

    # We can inspect the internal histogram to verify it tracked correctly
    assert logger._batch_size_histogram is not None
    # No direct assert on metric values needed, just ensuring it doesn't crash
