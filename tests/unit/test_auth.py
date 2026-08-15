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

from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.testclient import TestClient

import litserve.server
from litserve import LitAPI, LitServer
from litserve.utils import wrap_litserve_start


class SimpleAuthedLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if auth.scheme != "Bearer" or auth.credentials != "1234":
            raise HTTPException(status_code=401, detail="Bad token")


def test_authorized_custom():
    server = LitServer(SimpleAuthedLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", headers={"Authorization": "Bearer 1234"}, json={"input": 4.0})
        assert response.status_code == 200


def test_not_authorized_custom():
    server = LitServer(SimpleAuthedLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", headers={"Authorization": "Bearer wrong"}, json={"input": 4.0})
        assert response.status_code == 401


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: Request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output) -> Response:
        return {"output": output}


def test_authorized_api_key():
    litserve.server.LIT_SERVER_API_KEY = "abcd"
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", headers={"X-API-Key": "abcd"}, json={"input": 4.0})
        assert response.status_code == 200

    litserve.server.LIT_SERVER_API_KEY = None


def test_not_authorized_api_key():
    litserve.server.LIT_SERVER_API_KEY = "abcd"
    server = LitServer(SimpleLitAPI(), accelerator="cpu", devices=1, workers_per_device=1)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        response = client.post("/predict", headers={"X-API-Key": "wrong"}, json={"input": 4.0})
        assert response.status_code == 401

    litserve.server.LIT_SERVER_API_KEY = None


# --- Multi-API auth tests ---


class TokenAuthedLitAPI(SimpleLitAPI):
    """SimpleLitAPI with Bearer token auth parameterized by token."""

    def __init__(self, token: str, **kwargs):
        super().__init__(**kwargs)
        self._token = token

    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if auth.scheme != "Bearer" or auth.credentials != self._token:
            raise HTTPException(status_code=401, detail="Bad token")


def test_multi_api_second_api_auth_enforced():
    """Second API's authorize() must be checked, not just the first API's."""
    api1 = SimpleLitAPI(api_path="/no-auth")
    api2 = TokenAuthedLitAPI(token="alpha", api_path="/needs-auth")
    server = LitServer([api1, api2], accelerator="cpu", devices=1, workers_per_device=1)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # First API (no auth) should be accessible without credentials
        resp = client.post("/no-auth", json={"input": 5.0})
        assert resp.status_code == 200

        # Second API with wrong credentials should be rejected (send wrong token so
        # authorize() itself raises 401, avoiding FastAPI version differences where
        # a missing Authorization header raises 403 in older versions)
        resp = client.post("/needs-auth", headers={"Authorization": "Bearer wrong"}, json={"input": 5.0})
        assert resp.status_code == 401

        # Second API with correct credentials should succeed
        resp = client.post("/needs-auth", headers={"Authorization": "Bearer alpha"}, json={"input": 5.0})
        assert resp.status_code == 200


def test_multi_api_mixed_auth():
    """One API with auth, one without -- each should work independently."""
    api_authed = TokenAuthedLitAPI(token="alpha", api_path="/protected")
    api_open = SimpleLitAPI(api_path="/open")
    server = LitServer([api_authed, api_open], accelerator="cpu", devices=1, workers_per_device=1)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # Open API works without any auth
        resp = client.post("/open", json={"input": 10.0})
        assert resp.status_code == 200

        # Protected API rejects wrong credentials (use wrong token rather than no header
        # to avoid FastAPI version differences: missing header → 403 in older, 401 in newer)
        resp = client.post("/protected", headers={"Authorization": "Bearer wrong"}, json={"input": 10.0})
        assert resp.status_code == 401

        # Protected API accepts correct credentials
        resp = client.post("/protected", headers={"Authorization": "Bearer alpha"}, json={"input": 10.0})
        assert resp.status_code == 200


def test_multi_api_both_with_different_auth():
    """Two APIs with different auth tokens -- each checks its own authorize()."""
    api_a = TokenAuthedLitAPI(token="alpha", api_path="/api-a")
    api_b = TokenAuthedLitAPI(token="beta", api_path="/api-b")
    server = LitServer([api_a, api_b], accelerator="cpu", devices=1, workers_per_device=1)

    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        # API A with its own token succeeds
        resp = client.post("/api-a", headers={"Authorization": "Bearer alpha"}, json={"input": 2.0})
        assert resp.status_code == 200

        # API A with API B's token fails
        resp = client.post("/api-a", headers={"Authorization": "Bearer beta"}, json={"input": 2.0})
        assert resp.status_code == 401

        # API B with its own token succeeds
        resp = client.post("/api-b", headers={"Authorization": "Bearer beta"}, json={"input": 2.0})
        assert resp.status_code == 200

        # API B with API A's token fails
        resp = client.post("/api-b", headers={"Authorization": "Bearer alpha"}, json={"input": 2.0})
        assert resp.status_code == 401
