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
import copy
import multiprocessing as mp

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp

import litserve as ls
from litserve.middlewares import MaxSizeMiddleware, RequestCountMiddleware
from litserve.utils import wrap_litserve_start


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, length: int) -> None:
        self.app = app
        self.length = length
        super().__init__(app)

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Request-Id"] = "0" * self.length
        return response


def test_custom_middleware():
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=[(RequestIdMiddleware, {"length": 5})])
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0})
        assert response.status_code == 200, f"Expected response to be 200 but got {response.status_code}"
        assert response.json() == {"output": 16.0}, "server didn't return expected output"
        assert response.headers["X-Request-Id"] == "00000"


def test_starlette_middlewares():
    middlewares = [
        (
            TrustedHostMiddleware,
            {
                "allowed_hosts": ["localhost", "127.0.0.1"],
            },
        ),
        HTTPSRedirectMiddleware,
    ]
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=middlewares)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", json={"input": 4.0}, headers={"Host": "localhost"})
        assert response.status_code == 200, f"Expected response to be 200 but got {response.status_code}"
        assert response.json() == {"output": 16.0}, "server didn't return expected output"

        response = client.post("/predict", json={"input": 4.0}, headers={"Host": "not-trusted-host"})
        assert response.status_code == 400, f"Expected response to be 400 but got {response.status_code}"


def test_middlewares_inputs():
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=[])
    assert len(server.middlewares) == 1, "Default middleware should be present"

    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=[], max_payload_size=1000)
    assert len(server.middlewares) == 2, "Default middleware should be present"

    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=None)
    assert len(server.middlewares) == 1, "Default middleware should be present"

    with pytest.raises(ValueError, match="middlewares must be a list of tuples"):
        ls.LitServer(ls.test_examples.SimpleLitAPI(), middlewares=(RequestIdMiddleware, {"length": 5}))


@pytest.mark.parametrize(
    ("max_payload_size", "expected"),
    [
        (1000, 1000),
        (1048576.0, 1048576),
        ("500KB", 500 * 1024),
        ("2MB", 2 * 1024**2),
        ("1GB", 1024**3),
        ("1.5mb", int(1.5 * 1024**2)),
        (" 2 MB ", 2 * 1024**2),
        ("10B", 10),
    ],
)
def test_max_payload_size_parsed_at_init(max_payload_size, expected):
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), max_payload_size=max_payload_size)
    assert server.max_payload_size == expected
    assert (MaxSizeMiddleware, {"max_size": expected}) in server.middlewares


def test_max_payload_size_none_adds_no_middleware():
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), max_payload_size=None)
    assert server.max_payload_size is None
    assert all(middleware is not MaxSizeMiddleware for middleware, _ in server.middlewares)


@pytest.mark.parametrize(
    "max_payload_size", ["abc", "MB", "2XB", "", "-1MB", -1, 0, "0MB", True, [1], float("inf"), float("nan")]
)
def test_max_payload_size_invalid(max_payload_size):
    with pytest.raises(ValueError, match="max_payload_size"):
        ls.LitServer(ls.test_examples.SimpleLitAPI(), max_payload_size=max_payload_size)


def test_max_payload_size_string_enforced_on_requests():
    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), max_payload_size="1KB")
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        assert client.post("/predict", json={"input": 4.0}).status_code == 200
        assert client.post("/predict", json={"input": "x" * 2048}).status_code == 413


def test_middleware_multiple_initialization():
    api1 = ls.test_examples.SimpleLitAPI(api_path="/api1")
    api2 = ls.test_examples.SimpleLitAPI(api_path="/api2")
    api3 = ls.test_examples.SimpleLitAPI(api_path="/api3")
    api4 = ls.test_examples.SimpleLitAPI(api_path="/api4")

    server = ls.LitServer([api1, api2, api3, api4])
    assert len(server.app.user_middleware) == 1, "Each middleware should be initialized only once for `n` LitAPIs"


def test_track_requests_middleware_isolation():
    """Test that _prepare_app_run doesn't modify the original app's middleware list."""
    lit_api = ls.test_examples.SimpleLitAPI()
    server = ls.LitServer(lit_api, track_requests=True)

    # Store original middleware state
    original_middleware = server.app.user_middleware.copy()

    # Create app copies (simulating multiple API servers)
    app_copies = [copy.copy(server.app) for _ in range(3)]

    # Call _prepare_app_run on each copy
    for app_copy in app_copies:
        server._prepare_app_run(app_copy)

    # Verify each copy has RequestCountMiddleware added
    for app_copy in app_copies:
        assert len(app_copy.user_middleware) == len(original_middleware) + 1
        # Check that RequestCountMiddleware is added
        assert any(mw.cls is RequestCountMiddleware for mw in app_copy.user_middleware), (
            "RequestCountMiddleware not found in middleware list"
        )


def test_active_requests_reset_after_error():
    """A request that fails should not stay counted in `LitServer.active_requests`."""
    in_flight = []

    class FailingMiddleware(BaseHTTPMiddleware):
        """Fails a request downstream of RequestCountMiddleware, which wraps all user middlewares."""

        async def dispatch(self, request, call_next):
            await call_next(request)
            in_flight.append(server.active_requests)
            raise RuntimeError("request failed")

    server = ls.LitServer(ls.test_examples.SimpleLitAPI(), track_requests=True, middlewares=[FailingMiddleware])
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        with pytest.raises(RuntimeError, match="request failed"):
            client.post("/predict", json={"input": 4.0})

        assert in_flight == [1], "request should be counted while it is in flight"
        assert server.active_requests == 0, "failed request should not stay in the active request count"


@pytest.mark.parametrize(
    ("server_kwargs", "internal_paths"),
    [
        ({}, ["/", "/health", "/info"]),
        ({"healthcheck_path": "/healthz", "info_path": "/details"}, ["/", "/healthz", "/details"]),
    ],
)
def test_internal_paths_not_counted_in_active_requests(server_kwargs, internal_paths):
    """Internal endpoints, on default or custom paths, should not count as active requests."""
    in_flight = {}

    class RecordingMiddleware(BaseHTTPMiddleware):
        """Runs inside RequestCountMiddleware, so it sees the count while the request is in flight."""

        async def dispatch(self, request, call_next):
            in_flight[request.url.path] = server.active_requests
            return await call_next(request)

    server = ls.LitServer(
        ls.test_examples.SimpleLitAPI(), track_requests=True, middlewares=[RecordingMiddleware], **server_kwargs
    )
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        for path in internal_paths:
            client.get(path)  # status is irrelevant here, e.g. health is 503 without ready workers
        assert client.post("/predict", json={"input": 4.0}).status_code == 200

        assert in_flight == {**dict.fromkeys(internal_paths, 0), "/predict": 1}
        assert server.active_requests == 0


@pytest.mark.asyncio
async def test_request_count_middleware_excluded_paths():
    """Default excluded paths are skipped, also when uvicorn prefixes scope["path"] with root_path."""
    counter = mp.Value("i", 0, lock=True)
    in_flight = {}

    async def app(scope, receive, send):
        in_flight[scope["path"]] = counter.value

    middleware = RequestCountMiddleware(app, active_counter=counter)

    async def call(path, root_path="/api"):
        await middleware({"type": "http", "path": path, "root_path": root_path}, None, None)

    await call("/health", root_path="")
    await call("/predict", root_path="")
    await call("/api/health")
    await call("/api")
    await call("/api/predict")
    await call("/apiary/health")
    assert in_flight == {
        "/health": 0,
        "/predict": 1,
        "/api/health": 0,
        "/api": 0,
        "/api/predict": 1,
        "/apiary/health": 1,
    }
    assert counter.value == 0
