import asyncio
from multiprocessing import Manager
from typing import Any, Optional

from litserve.transport.base import MessageTransport


class MPQueueTransport(MessageTransport):
    def __init__(self, manager: Manager, num_consumers: int):
        self._queues = [manager.Queue() for _ in range(num_consumers)]

    def send(self, item: Any, consumer_id: int) -> None:
        return self._queues[consumer_id].put(item)

    async def areceive(self, consumer_id: int, timeout: Optional[float] = None) -> dict:
        return await asyncio.to_thread(self._queues[consumer_id].get, timeout=timeout, block=True)

    def close(self) -> None:
        pass
