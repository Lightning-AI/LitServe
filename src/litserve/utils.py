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
from contextlib import contextmanager
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import HTTPException

if TYPE_CHECKING:
    from litserve.server import LitServer

logger = logging.getLogger(__name__)


class LitAPIStatus:
    OK = "OK"
    ERROR = "ERROR"
    FINISH_STREAMING = "FINISH_STREAMING"


class PickleableHTTPException(HTTPException):
    @staticmethod
    def from_exception(exc: HTTPException):
        status_code = exc.status_code
        detail = exc.detail
        return PickleableHTTPException(status_code, detail)

    def __reduce__(self):
        return (HTTPException, (self.status_code, self.detail))


def dump_exception(exception):
    if isinstance(exception, HTTPException):
        exception = PickleableHTTPException.from_exception(exception)
    return pickle.dumps(exception)


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
    server._prepare_app_run(server.app)
    yield server
    for p in processes:
        p.terminate()
    manager.shutdown()


async def call_after_stream(streamer: AsyncIterator, callback, *args, **kwargs):
    try:
        async for item in streamer:
            yield item
    except Exception as e:
        logger.exception(f"Error in streamer: {e}")
    finally:
        callback(*args, **kwargs)
