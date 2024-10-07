import multiprocessing
from typing import Optional

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from starlette.types import ASGIApp, Message, Receive, Scope, Send


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


class RequestCountMiddleware(BaseHTTPMiddleware):
    """Adds a header to the response with the number of active requests."""

    def __init__(self, app: ASGIApp, active_counter: multiprocessing.Value) -> None:
        self.app = app
        self._active_requests = active_counter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        lock = self._active_requests.get_lock()
        with lock:
            self._active_requests.value += 1
        await self.app(scope, receive, send)
        with lock:
            self._active_requests.value -= 1
