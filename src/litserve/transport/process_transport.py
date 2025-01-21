import asyncio
from multiprocessing import Manager, Queue
from typing import Any, List, Optional

from litserve.transport.base import MessageTransport


class MPQueueTransport(MessageTransport):
    def __init__(self, manager: Manager, queues: List[Queue]):
        self._queues = queues

    def send(self, item: Any, consumer_id: int) -> None:
        return self._queues[consumer_id].put(item)

    async def areceive(self, consumer_id: int, timeout: Optional[float] = None) -> dict:
        return await asyncio.to_thread(self._queues[consumer_id].get, timeout=timeout, block=True)

    def close(self) -> None:
        pass

    def __reduce__(self):
        return (MPQueueTransport, (None, self._queues))
