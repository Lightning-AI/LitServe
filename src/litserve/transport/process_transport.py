import asyncio
from contextlib import suppress
from multiprocessing import Manager, Queue
from typing import Any, List, Optional

from litserve.transport.base import MessageTransport


class MPQueueTransport(MessageTransport):
    def __init__(self, manager: Manager, queues: List[Queue]):
        self._queues = queues
        self._closed = False

    def send(self, item: Any, consumer_id: int) -> None:
        return self._queues[consumer_id].put(item)

    async def areceive(self, consumer_id: int, timeout: Optional[float] = None, block: bool = True) -> dict:
        if self._closed:
            raise asyncio.CancelledError("Transport closed")

        actual_timeout = 1 if timeout is None else min(timeout, 1)

        try:
            return await asyncio.to_thread(self._queues[consumer_id].get, timeout=actual_timeout, block=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._closed:
                raise asyncio.CancelledError("Transport closed")
            if timeout is not None and timeout <= actual_timeout:
                raise
            return None

    def close(self) -> None:
        # Mark the transport as closed
        self._closed = True

        # Put sentinel values in the queues as a backup mechanism
        for queue in self._queues:
            with suppress(Exception):
                queue.put(None)

    def __reduce__(self):
        return (MPQueueTransport, (None, self._queues))
