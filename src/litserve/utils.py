import asyncio
import logging
import pickle
from typing import Coroutine, Optional
import uuid

from fastapi import HTTPException

logger = logging.getLogger(__name__)


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
        pickle.loads(response)
        raise HTTPException(500, "Internal Server Error")
    except pickle.PickleError:
        logger.exception(
            f"main process failed to load the exception from the parallel worker process. "
            f"{response} couldn't be unpickled."
        )
