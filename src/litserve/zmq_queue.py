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

logging.basicConfig(level=logging.INFO)
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
            frontend.close()
            backend.close()
            context.term()

    def stop(self):
        """Stop the broker."""
        self._running = False
        if self._worker:
            self._worker.join()


class Producer:
    """Producer class for sending messages to a specific consumer."""

    def __init__(self, consumer_id: int, address: str = None):
        self.consumer_id = consumer_id
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.connect(address)

    def put(self, item: Any) -> None:
        """Send an item to the consumer."""
        try:
            # Serialize the item using pickle
            pickled_item = pickle.dumps(item)
            message = f"{self.consumer_id}|".encode() + pickled_item
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


class Consumer(BaseConsumer):
    """Synchronous consumer class for receiving messages."""

    def _setup_socket(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(self.address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, str(self.consumer_id))

    def get(self, timeout: Optional[int] = None) -> Any:
        """Get an item from the queue."""
        if timeout is not None and not self._socket.poll(timeout * 1000):
            raise Empty

        try:
            message = self._socket.recv()
            return self._parse_message(message)
        except zmq.ZMQError:
            raise Empty


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

    async def aclose(self) -> None:
        """Clean up resources asynchronously."""
        if self._socket:
            self._socket.close()
        if self._context:
            await self._context.term()


# Example usage
def example_usage():
    # Start the broker
    broker = Broker(use_process=False)
    broker.start()

    # Create producer and sync consumer
    producer = Producer(consumer_id=0, address=broker.backend_address)
    consumer = Consumer(consumer_id=0, address=broker.frontend_address)
    time.sleep(2)  # Give the producer and consumer time to connect
    try:
        # Send some complex Python objects
        producer.put({"hello": "world", "data": [1, 2, 3]})
        producer.put(("tuple", 123, {"nested": True}))

        # Receive messages synchronously
        try:
            data = consumer.get(timeout=1.0)
            print(f"Received sync: {data}")  # Will print the dict

            data = consumer.get(timeout=1.0)
            print(f"Received sync: {data}")  # Will print the tuple

        except Empty:
            print("No data available")

    finally:
        producer.close()
        consumer.close()
        broker.stop()


async def async_example():
    # Start the broker
    broker = Broker()
    broker.start()

    # Create producer and async consumer
    producer = Producer(consumer_id=0, address=broker.backend_address)
    consumer = AsyncConsumer(consumer_id=0, address=broker.frontend_address)

    await asyncio.sleep(0.1)  # Give time to connect

    try:
        # Send some messages
        producer.put("Hello")
        producer.put("World")

        # Receive messages asynchronously
        try:
            data = await consumer.get(timeout=1.0)
            print(f"Received async: {data}")

            data = await consumer.get(timeout=1.0)
            print(f"Received async: {data}")

        except Empty:
            print("No data available")

    finally:
        producer.close()
        await consumer.aclose()
        broker.stop()


if __name__ == "__main__":
    # Run both examples
    print("Running sync example:")
    example_usage()

    print("\nRunning async example:")
    asyncio.run(async_example())
