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

import asyncio
import logging
import pickle
from typing import Optional, Union
from contextlib import contextmanager
from typing import TYPE_CHECKING

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from litserve.server import LitServer

logger = logging.getLogger(__name__)


class LitAPIStatus:
    OK = "OK"
    ERROR = "ERROR"
    FINISH_STREAMING = "FINISH_STREAMING"


def load_and_raise(response: Union[bytes, Exception]) -> None:
    try:
        if isinstance(response, bytes):
            exception = pickle.loads(response)
        else:
            exception = response
        raise exception
    except pickle.PickleError:
        logger.exception(
            f"Main process failed to load the exception from the parallel worker process. "
            f"{response} couldn't be unpickled."
        )
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error while processing the response: {e}"
        )
        raise


async def azip(*async_iterables):
    iterators = [ait.__aiter__() for ait in async_iterables]
    while True:
        results = await asyncio.gather(*(ait.__anext__() for ait in iterators), return_exceptions=True)
        if any(isinstance(result, StopAsyncIteration) for result in results):
            break
        yield tuple(results)


@contextmanager
def wrap_litserve_start(server: "LitServer"):
    server.app.response_queue_id = 0
    if server.lit_spec:
        server.lit_spec.response_queue_id = 0
    manager, processes = server.launch_inference_worker(num_uvicorn_servers=1)
    try:
        yield server
    finally:
        for p in processes:
            p.terminate()
            p.join()  # Ensure the process has fully exited
        manager.shutdown()


class MaxSizeMiddleware(BaseHTTPMiddleware):
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
            try:
                message = await receive()
            except Exception as e:
                raise HTTPException(500, f"Error receiving message: {str(e)}")

            chunk_size = len(message.get("body", b""))
            total_size += chunk_size
            if self.max_size is not None and total_size > self.max_size:
                logger.warning(f"Payload size exceeded: {total_size} bytes (max allowed: {self.max_size} bytes)")
                raise HTTPException(413, "Payload too large")
            return message

        await self.app(scope, rcv, send)
