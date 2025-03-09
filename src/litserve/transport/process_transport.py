import asyncio
from contextlib import suppress
from multiprocessing import Manager, Queue
from typing import Any, List, Optional

from litserve.transport.base import MessageTransport


class MPQueueTransport(MessageTransport):
    def __init__(self, manager: Manager, queues: List[Queue]):
        self._queues = queues
        self._closed = False
        self._terminate_event = asyncio.Event()
        self._mp_terminate_event = manager.Event() if manager else None

    def send(self, item: Any, consumer_id: int) -> None:
        if self._closed or (self._mp_terminate_event and self._mp_terminate_event.is_set()):
            return None
        return self._queues[consumer_id].put(item)

    async def areceive(self, consumer_id: int, timeout: Optional[float] = None, block: bool = True) -> dict:
        # Check if we're already closed
        if self._closed or (self._mp_terminate_event and self._mp_terminate_event.is_set()):
            raise asyncio.CancelledError("Transport closed")

        # Use a short timeout to periodically check the MP event
        # This is a compromise - we need some timeout to check for termination
        actual_timeout = 0.1 if timeout is None else min(timeout, 0.1)

        while not self._closed and not (self._mp_terminate_event and self._mp_terminate_event.is_set()):
            try:
                return await asyncio.to_thread(self._queues[consumer_id].get, timeout=actual_timeout, block=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                if self._closed or (self._mp_terminate_event and self._mp_terminate_event.is_set()):
                    raise asyncio.CancelledError("Transport closed")
                if timeout is not None:
                    timeout -= actual_timeout
                    if timeout <= 0:
                        raise
                continue
        return None

    def close(self) -> None:
        # Mark the transport as closed
        self._closed = True

        # Set the multiprocessing event to signal across processes
        if self._mp_terminate_event:
            self._mp_terminate_event.set()

        # Set the asyncio event to signal within this process
        self._terminate_event.set()

        # Put sentinel values in the queues as a backup mechanism
        for queue in self._queues:
            with suppress(Exception):
                queue.put(None)

    def __reduce__(self):
        # We don't pickle the asyncio event, only the MP event and queues
        return (MPQueueTransport, (None, self._queues))
