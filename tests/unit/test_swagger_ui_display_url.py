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
import contextlib
from unittest.mock import MagicMock, patch

import pytest

from litserve import LitAPI, LitServer


class MinimalLitAPI(LitAPI):
    def setup(self, device):
        pass

    def decode_request(self, request):
        return request

    def predict(self, x):
        return x

    def encode_response(self, output):
        return output


@pytest.mark.parametrize(
    ("host", "display_host"),
    [
        ("0.0.0.0", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("::", "localhost"),
    ],
)
@patch("builtins.print")
@patch("litserve.server.uvicorn")
def test_swagger_ui_message_uses_browser_reachable_host(mock_uvicorn, mock_print, mock_manager, host, display_host):
    server = LitServer(MinimalLitAPI(), restart_workers=False)
    server.verify_worker_status = MagicMock()

    with (
        patch("litserve.server.mp.Manager", return_value=mock_manager),
        contextlib.suppress(Exception),
    ):
        server._monitor_workers = False
        server.run(host=host, port=8000, generate_client_file=False)

    mock_print.assert_called_with(f"Swagger UI is available at http://{display_host}:8000/docs")


@patch("builtins.print")
@patch("litserve.server.uvicorn")
def test_swagger_ui_message_is_hidden_when_openapi_url_is_disabled(mock_uvicorn, mock_print, mock_manager):
    server = LitServer(MinimalLitAPI(), disable_openapi_url=True, restart_workers=False)
    server.verify_worker_status = MagicMock()

    with (
        patch("litserve.server.mp.Manager", return_value=mock_manager),
        contextlib.suppress(Exception),
    ):
        server._monitor_workers = False
        server.run(port=8000, generate_client_file=False)

    mock_print.assert_not_called()
