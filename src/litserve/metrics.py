# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

from litserve.loggers import Logger
from litserve.utils import is_package_installed

logger = logging.getLogger(__name__)

_PROMETHEUS_AVAILABLE = is_package_installed("prometheus_client")


def _check_prometheus_available():
    if not _PROMETHEUS_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "prometheus_client is not installed. Please install it with `pip install litserve[prometheus]` "
            "to use PrometheusLogger."
        )


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP request metrics for Prometheus."""

    def __init__(self, app):
        super().__init__(app)
        _check_prometheus_available()

        # Use singleton pattern to avoid hitting private registry internals or re-creating metrics
        if not hasattr(PrometheusMiddleware, "_metrics_initialized"):
            from prometheus_client import Counter, Histogram

            PrometheusMiddleware.http_requests_total = Counter(
                "litserve_http_requests_total", "Total requests", ["method", "endpoint", "status"]
            )
            PrometheusMiddleware.http_request_duration_seconds = Histogram(
                "litserve_http_request_duration_seconds", "Request latency", ["endpoint"]
            )
            PrometheusMiddleware._metrics_initialized = True

    async def dispatch(self, request, call_next):
        if request.url.path in ["/metrics", "/health", "/info"]:
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        PrometheusMiddleware.http_request_duration_seconds.labels(endpoint=request.url.path).observe(duration)
        PrometheusMiddleware.http_requests_total.labels(
            method=request.method, endpoint=request.url.path, status=response.status_code
        ).inc()
        return response


class PrometheusLogger(Logger):
    """A built-in logger that exposes Prometheus metrics in a multiprocess environment."""

    def __init__(self):
        super().__init__()
        _check_prometheus_available()

        async def lazy_asgi_app(scope, receive, send):
            if not hasattr(self, "_asgi_app"):
                from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                self._asgi_app = make_asgi_app(registry=registry)
            await self._asgi_app(scope, receive, send)

        self.mount("/metrics", lazy_asgi_app)

        # We don't initialize the Histogram here because PROMETHEUS_MULTIPROC_DIR
        # is not guaranteed to be set yet. It will be initialized lazily.
        self._batch_size_histogram = None

    def process(self, key, value):
        """Handle self.log() calls from LitAPI workers."""
        if self._batch_size_histogram is None:
            from prometheus_client import Histogram

            # By the time process() is called, the server has started and set PROMETHEUS_MULTIPROC_DIR
            self._batch_size_histogram = Histogram("litserve_batch_size", "Inference batch size")

        if key == "batch_size":
            self._batch_size_histogram.observe(value)

    def close(self) -> None:
        """Clean up temporary multiprocess metrics directory on shutdown."""
        import os

        if getattr(self, "_metrics_dir", None) and os.path.exists(self._metrics_dir):
            import shutil

            shutil.rmtree(self._metrics_dir, ignore_errors=True)
            self._metrics_dir = None


def setup_prometheus(loggers, middlewares):
    """Helper to initialize Prometheus multiprocess directory and middleware if PrometheusLogger is used."""
    if loggers is None:
        return None

    _loggers_list = loggers if isinstance(loggers, list) else [loggers]
    prom_loggers = [logger for logger in _loggers_list if isinstance(logger, PrometheusLogger)]
    if not prom_loggers:
        return None

    _check_prometheus_available()

    import os
    import tempfile

    import prometheus_client.values

    metrics_dir = tempfile.mkdtemp(prefix="litserve_prom_")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
    prometheus_client.values.ValueClass = prometheus_client.values.MultiProcessValue()
    middlewares.append(PrometheusMiddleware)

    for logger in prom_loggers:
        logger._metrics_dir = metrics_dir

    return metrics_dir
