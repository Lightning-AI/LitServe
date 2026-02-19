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
import uuid
from queue import Empty
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import iceoryx2
except ImportError:
    raise ImportError("iceoryx2 is not installed. Install with: pip install litserve[iceoryx2]")

# Constants
DEFAULT_SERVICE_PREFIX = "litserve"
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB default
CONNECTION_TIMEOUT = 5.0


class Producer:
    """Producer class for sending messages via iceoryx2 Publisher."""

    def __init__(self, service_name: str, node=None):
        self.service_name = service_name
        self._node = node or self._create_node()
        self._publisher = None
        self._setup_publisher()

    def _create_node(self):
        """Create iceoryx2 node."""
        return iceoryx2.Node()

    def _setup_publisher(self):
        """Create publisher for the service."""
        service = self._node.service_builder(self.service_name).publish_subscribe().open_or_create()
        self._publisher = service.publisher_builder().create()

    def wait_for_subscribers(self, timeout: float = 1.0) -> bool:
        """Wait for at least one subscriber to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if subscribers are ready, False if timeout occurred

        """
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Send a ping message to consumer 0 (special system messages)
            try:
                self.put(b"__ping__", consumer_id=0)
                time.sleep(0.1)  # Give time for subscription to propagate
                return True
            except (RuntimeError, AttributeError):
                continue
        return False

    def put(self, item: Any, consumer_id: int) -> None:
        """Send an item to a specific consumer."""
        try:
            pickled_item = pickle.dumps(item)
            message = f"{consumer_id}|".encode() + pickled_item
            # Send via iceoryx2 publisher
            sample = self._publisher.loan_slice(len(message))
            sample.copy_from_slice(message)
            sample.send()
        except pickle.PickleError as e:
            logger.error(f"Error serializing item: {e}")
            raise
        except Exception as e:
            logger.error(f"Error sending item: {e}")
            raise

    def close(self) -> None:
        """Clean up resources."""
        if self._publisher:
            del self._publisher
        if self._node:
            del self._node


class AsyncConsumer:
    """Async consumer for receiving messages via iceoryx2 Subscriber."""

    def __init__(self, service_name: str, consumer_id: int, node=None):
        self.service_name = service_name
        self.consumer_id = consumer_id
        self._node = node or self._create_node()
        self._subscriber = None
        self._setup_subscription()

    def _create_node(self):
        """Create iceoryx2 node."""
        return iceoryx2.Node()

    def _setup_subscription(self):
        """Create subscriber for the service."""
        service = self._node.service_builder(self.service_name).publish_subscribe().open_or_create()
        self._subscriber = service.subscriber_builder().create()

    async def get(self, timeout: Optional[float] = None) -> Any:
        """Get an item from the queue asynchronously."""
        try:
            if timeout is not None:
                message = await asyncio.wait_for(self._receive_message(), timeout)
            else:
                message = await self._receive_message()
            return self._parse_message(message)
        except asyncio.TimeoutError:
            raise Empty

    async def _receive_message(self) -> bytes:
        """Receive a message from the subscriber."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_receive)

    def _sync_receive(self) -> bytes:
        """Synchronously receive a message."""
        sample = self._subscriber.receive()
        data = bytes(sample.payload)
        return data

    def _parse_message(self, message: bytes) -> Any:
        """Parse message, filtering by consumer_id."""
        try:
            consumer_id_bytes, pickled_data = message.split(b"|", 1)
            received_consumer_id = int(consumer_id_bytes.decode())
            if received_consumer_id != self.consumer_id:
                raise Empty(f"Message for consumer {received_consumer_id}, not {self.consumer_id}")
            return pickle.loads(pickled_data)
        except (ValueError, pickle.PickleError) as e:
            logger.error(f"Error parsing message: {e}")
            raise

    def close(self) -> None:
        """Clean up resources."""
        if self._subscriber:
            del self._subscriber
        if self._node:
            del self._node


def generate_service_name(prefix: str, suffix: Optional[str] = None) -> str:
    """Generate unique service name for iceoryx2."""
    unique_id = suffix or str(uuid.uuid4())[:8]
    return f"{prefix}-{unique_id}"


def create_iceoryx2_service_names() -> tuple[str, str]:
    """Generate unique service names for Iceoryx2Transport."""
    suffix = str(uuid.uuid4())[:8]
    frontend = generate_service_name("litserve-frontend", suffix)
    backend = generate_service_name("litserve-backend", suffix)
    return frontend, backend
