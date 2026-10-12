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
import multiprocessing
from collections.abc import Collection
from typing import Optional

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class MaxSizeMiddleware(BaseHTTPMiddleware):
    """Rejects requests with a payload that is too large."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_size: Optional[int] = None,
    ) -> None:
        self.app = app
        self.max_size = max_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        total_size = 0

        async def rcv() -> Message:
            nonlocal total_size
            message = await receive()
            chunk_size = len(message.get("body", b""))
            total_size += chunk_size
            if self.max_size is not None and total_size > self.max_size:
                raise HTTPException(413, "Payload too large")
            return message

        await self.app(scope, rcv, send)


def _route_path(scope: Scope) -> str:
    """Return ``scope["path"]`` without ``root_path``, which uvicorn prepends to it."""
    path, root_path = scope["path"], scope.get("root_path", "")
    if root_path and (path == root_path or path.startswith(root_path + "/")):
        return path[len(root_path) :] or "/"
    return path


class RequestCountMiddleware(BaseHTTPMiddleware):
    """Tracks in-flight requests in ``active_counter``, skipping ``excluded_paths``."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        active_counter: multiprocessing.Value,
        excluded_paths: Collection[str] = ("/", "/health", "/metrics"),
    ) -> None:
        self.app = app
        self.active_counter = active_counter
        self.excluded_paths = frozenset(excluded_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or _route_path(scope) in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        with self.active_counter.get_lock():
            self.active_counter.value += 1
        try:
            await self.app(scope, receive, send)
        finally:
            with self.active_counter.get_lock():
                self.active_counter.value -= 1
