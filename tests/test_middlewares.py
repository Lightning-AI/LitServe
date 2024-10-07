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
import pytest
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp

import litserve as ls
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
