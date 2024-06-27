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
from typing import Coroutine, Optional
import uuid
import time
import asyncio
from multiprocessing.connection import Connection
from functools import wraps

from fastapi import HTTPException


logger = logging.getLogger(__name__)

# Set up second logger
server_logger = logging.getLogger('LitServer')
server_logger.setLevel(logging.INFO)
file_handler2 = logging.FileHandler('logs/LitServer_log.log')
formatter2 = logging.Formatter("%(asctime)s,%(name)s,%(message)s",)
file_handler2.setFormatter(formatter2)
server_logger.addHandler(file_handler2)


class LitAPIStatus:
    OK = "OK"
    ERROR = "ERROR"
    FINISH_STREAMING = "FINISH_STREAMING"


async def wait_for_queue_timeout(coro: Coroutine, timeout: Optional[float], uid: uuid.UUID, request_buffer: dict):
    if timeout == -1 or timeout is False:
        return await coro

    task = asyncio.create_task(coro)
    shield = asyncio.shield(task)
    try:
        return await asyncio.wait_for(shield, timeout)
    except asyncio.TimeoutError:
        if uid in request_buffer:
            logger.error(
                f"Request was waiting in the queue for too long ({timeout} seconds) and has been timed out. "
                "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
            )
            raise HTTPException(504, "Request timed out")
        return await task


def load_and_raise(response):
    try:
        exception = pickle.loads(response)
        raise exception
    except pickle.PickleError:
        logger.exception(
            f"main process failed to load the exception from the parallel worker process. "
            f"{response} couldn't be unpickled."
        )
        raise


async def azip(*async_iterables):
    iterators = [ait.__aiter__() for ait in async_iterables]
    while True:
        results = await asyncio.gather(*(ait.__anext__() for ait in iterators), return_exceptions=True)
        if any(isinstance(result, StopAsyncIteration) for result in results):
            break
        yield tuple(results)



def log_time(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            server_logger.info(f"{func.__name__} (ms), {elapsed_time:.3f}")
            return result
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            server_logger.info(f"{func.__name__} (ms), {elapsed_time:.2f}")
            return result
    return wrapper

class Timing:
    def __init__(self, name:str=None):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = (self.end_time - self.start_time) * 1000  # Convert to milliseconds
        server_logger.info(f"{self.name} (ms), {self.elapsed_time:.2f}")


def pipe_send(conn:Connection, data):
    with Timing("pipe_send"):
        conn.send(data)

def pipe_read(conn:Connection):
    with Timing("pipe_read"):
        return conn.recv()
