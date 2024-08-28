import litserve as ls

from starlette.types import ASGIApp
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, length: int) -> None:
        self.app = app
        self.length = length
        super().__init__(app)

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Request-Id"] = "0" * self.length
        return response


if __name__ == "__main__":
    api = ls.examples.SimpleLitAPI()
    server = ls.LitServer(api, middlewares=[(RequestIdMiddleware, {"length": 5})])
    server.run(port=8000)
