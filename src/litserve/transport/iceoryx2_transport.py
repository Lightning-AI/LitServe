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
from typing import Any, Literal, Optional, Union

from litserve.transport.base import MessageTransport
from litserve.transport.iceoryx2_queue import AsyncConsumer, Producer


class Iceoryx2Transport(MessageTransport):
    """Iceoryx2-based transport using zero-copy shared memory."""

    def __init__(self, frontend_service: str, backend_service: str):
        self.frontend_service = frontend_service
        self.backend_service = backend_service
        self._iceoryx2: Union[Producer, AsyncConsumer, None] = None

    def setup(self, operation: Literal["pub", "sub"], consumer_id: Optional[int] = None) -> None:
        """Setup in subprocess.

        Use 'pub' for publisher, 'sub' for subscriber.

        """
        if operation == "pub":
            self._iceoryx2 = Producer(service_name=self.backend_service)
            self._iceoryx2.wait_for_subscribers()
        elif operation == "sub":
            if consumer_id is None:
                raise ValueError("consumer_id required for subscriber setup")
            self._iceoryx2 = AsyncConsumer(service_name=self.frontend_service, consumer_id=consumer_id)
        else:
            raise ValueError(f"Invalid operation {operation}")

    def send(self, item: Any, consumer_id: int) -> None:
        """Send message to consumer."""
        if self._iceoryx2 is None:
            self.setup("pub")
        return self._iceoryx2.put(item, consumer_id)

    async def areceive(self, timeout: Optional[int] = None, consumer_id: Optional[int] = None) -> dict:
        """Receive message from publisher."""
        if self._iceoryx2 is None:
            if consumer_id is None:
                raise ValueError("consumer_id required for first receive")
            self.setup("sub", consumer_id)
        return await self._iceoryx2.get(timeout=timeout)

    def close(self, **kwargs) -> None:
        """Clean up resources."""
        if self._iceoryx2:
            self._iceoryx2.close()
        else:
            raise ValueError("Iceoryx2 not initialized")

    def __reduce__(self):
        """Allow pickling for multiprocessing."""
        return Iceoryx2Transport, (self.frontend_service, self.backend_service)
