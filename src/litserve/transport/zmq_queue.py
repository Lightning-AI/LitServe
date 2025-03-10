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
import multiprocessing
import pickle
import threading
import time
from queue import Empty
from typing import Any, Optional

import zmq
import zmq.asyncio

from litserve.utils import generate_random_zmq_address

logger = logging.getLogger(__name__)


class Broker:
    """Message broker that routes messages between producers and consumers."""

    def __init__(self, use_process: bool = False):
        self.frontend_address = generate_random_zmq_address()
        self.backend_address = generate_random_zmq_address()
        self._running = False
        self._use_process = use_process
        self._worker = None

    def start(self):
        """Start the broker in a background thread or process."""
        self._running = True

        if self._use_process:
            self._worker = multiprocessing.Process(target=self._run)
        else:
            self._worker = threading.Thread(target=self._run)

        self._worker.daemon = True
        self._worker.start()
        logger.info(
            f"Broker started in {'process' if self._use_process else 'thread'} "
            f"on {self.frontend_address} (frontend) and {self.backend_address} (backend)"
        )
        time.sleep(0.1)  # Give the broker time to start

    def _run(self):
        """Main broker loop."""
        context = zmq.Context()
        try:
            frontend = context.socket(zmq.XPUB)
            frontend.bind(self.frontend_address)

            backend = context.socket(zmq.XSUB)
            backend.bind(self.backend_address)

            zmq.proxy(frontend, backend)
        except zmq.ZMQError as e:
            logger.error(f"Broker error: {e}")
        finally:
            frontend.close(linger=0)
            backend.close(linger=0)
            context.term()

    def stop(self):
        """Stop the broker."""
        self._running = False
        if self._worker:
            self._worker.join()


class Producer:
    """Producer class for sending messages to consumers."""

    def __init__(self, address: str = None):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.connect(address)

    def wait_for_subscribers(self, timeout: float = 1.0) -> bool:
        """Wait for at least one subscriber to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if subscribers are ready, False if timeout occurred

        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Send a ping message to consumer 0 (special system messages)
            try:
                self._socket.send(b"0|__ping__", zmq.NOBLOCK)
                time.sleep(0.1)  # Give time for subscription to propagate
                return True
            except zmq.ZMQError:
                continue
        return False

    def put(self, item: Any, consumer_id: int) -> None:
        """Send an item to a specific consumer."""
        try:
            pickled_item = pickle.dumps(item)
            message = f"{consumer_id}|".encode() + pickled_item
            self._socket.send(message)
        except zmq.ZMQError as e:
            logger.error(f"Error sending item: {e}")
            raise
        except pickle.PickleError as e:
            logger.error(f"Error serializing item: {e}")
            raise

    def close(self) -> None:
        """Clean up resources."""
        if self._socket:
            self._socket.close(linger=0)
        if self._context:
            self._context.term()


class BaseConsumer:
    """Base class for consumers."""

    def __init__(self, consumer_id: int, address: str):
        self.consumer_id = consumer_id
        self.address = address
        self._context = None
        self._socket = None
        self._setup_socket()

    def _setup_socket(self):
        """Setup ZMQ socket - to be implemented by subclasses"""
        raise NotImplementedError

    def _parse_message(self, message: bytes) -> Any:
        """Parse a message received from ZMQ."""
        try:
            consumer_id, pickled_data = message.split(b"|", 1)
            return pickle.loads(pickled_data)
        except pickle.PickleError as e:
            logger.error(f"Error deserializing message: {e}")
            raise

    def close(self) -> None:
        """Clean up resources."""
        if self._socket:
            self._socket.close(linger=0)
        if self._context:
            self._context.term()


class AsyncConsumer(BaseConsumer):
    """Async consumer class for receiving messages using asyncio."""

    def _setup_socket(self):
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(self.address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, str(self.consumer_id))

    async def get(self, timeout: Optional[float] = None) -> Any:
        """Get an item from the queue asynchronously."""
        try:
            if timeout is not None:
                message = await asyncio.wait_for(self._socket.recv(), timeout)
            else:
                message = await self._socket.recv()

            return self._parse_message(message)
        except asyncio.TimeoutError:
            raise Empty
        except zmq.ZMQError:
            raise Empty

    def close(self) -> None:
        """Clean up resources asynchronously."""
        if self._socket:
            self._socket.close(linger=0)
        if self._context:
            self._context.term()
