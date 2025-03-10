from typing import Any, Literal, Optional, Union

import zmq

from litserve.transport.base import MessageTransport
from litserve.transport.zmq_queue import AsyncConsumer, Producer


class ZMQTransport(MessageTransport):
    def __init__(self, backend_address: str, frontend_address):
        self.backend_address = backend_address
        self.frontend_address = frontend_address
        self._zmq: Union[Producer, AsyncConsumer, None] = None

    def setup(self, operation: Literal[zmq.SUB, zmq.PUB], consumer_id: Optional[int] = None) -> None:
        """Must be called in the subprocess to setup the ZMQ transport."""
        if operation == zmq.PUB:
            self._zmq = Producer(address=self.backend_address)
            self._zmq.wait_for_subscribers()
        elif operation == zmq.SUB:
            self._zmq = AsyncConsumer(consumer_id=consumer_id, address=self.frontend_address)
        else:
            ValueError(f"Invalid operation {operation}")

    def send(self, item: Any, consumer_id: int) -> None:
        if self._zmq is None:
            self.setup(zmq.PUB)
        return self._zmq.put(item, consumer_id)

    async def areceive(self, consumer_id: Optional[int] = None, timeout=None) -> dict:
        if self._zmq is None:
            self.setup(zmq.SUB, consumer_id)
        return await self._zmq.get(timeout=timeout)

    def close(self) -> None:
        if self._zmq:
            self._zmq.close()
        else:
            raise ValueError("ZMQ not initialized, make sure ZMQTransport.setup() is called.")

    def __reduce__(self):
        return ZMQTransport, (self.backend_address, self.frontend_address)
