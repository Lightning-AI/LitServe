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
import pickle
import sys
from queue import Empty
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import iceoryx2 modules only if available
try:
    from litserve.transport.iceoryx2_queue import (
        AsyncConsumer,
        Producer,
        create_iceoryx2_service_names,
        generate_service_name,
    )
    ICEORYX2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ICEORYX2_AVAILABLE = False

# Skip all tests in this module if iceoryx2 is not available or Windows
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not ICEORYX2_AVAILABLE,
    reason="iceoryx2 not available or Windows platform"
)


@pytest.fixture
def mock_node():
    """Create a mock iceoryx2 Node."""
    with patch("litserve.transport.iceoryx2_queue.iceoryx2.Node") as mock_node_cls:
        node = Mock()
        mock_node_cls.return_value = node
        yield node


@pytest.fixture
def mock_publisher(mock_node):
    """Create a mock iceoryx2 Publisher."""
    publisher = Mock()
    mock_node.service_builder.return_value.publish_subscribe.return_value.open_or_create.return_value.publisher_builder.return_value.create.return_value = publisher

    # Mock the loan_slice/sample pattern
    sample = Mock()
    sample.copy_from_slice = Mock()
    sample.send = Mock()
    publisher.loan_slice = Mock(return_value=sample)

    return publisher


@pytest.fixture
def mock_subscriber(mock_node):
    """Create a mock iceoryx2 Subscriber."""
    subscriber = Mock()

    # Mock receive to return sample with payload
    sample = Mock()
    sample.payload = b"1|test_data"

    subscriber.receive = Mock(return_value=sample)
    mock_node.service_builder.return_value.publish_subscribe.return_value.open_or_create.return_value.subscriber_builder.return_value.create.return_value = subscriber

    return subscriber


def test_generate_service_name():
    """Test service name generation."""
    name = generate_service_name("test-prefix")
    assert name.startswith("test-prefix-")
    assert len(name.split("-")) == 3  # prefix, random part

    name_with_suffix = generate_service_name("test-prefix", "abc123")
    assert name_with_suffix == "test-prefix-abc123"


def test_create_iceoryx2_service_names():
    """Test paired service name creation."""
    frontend, backend = create_iceoryx2_service_names()

    assert frontend.startswith("litserve-frontend-")
    assert backend.startswith("litserve-backend-")

    # Suffixes should match
    frontend_suffix = frontend.split("-")[-1]
    backend_suffix = backend.split("-")[-1]
    assert frontend_suffix == backend_suffix


def test_producer_init(mock_node, mock_publisher):
    """Test Producer initialization."""
    producer = Producer(service_name="test-service")

    assert producer.service_name == "test-service"
    assert mock_node.service_builder.called


def test_producer_send(mock_node, mock_publisher):
    """Test Producer.send() method."""
    producer = Producer(service_name="test-service")

    # Test sending simple data
    producer.put("test_data", consumer_id=1)

    # Verify loan_slice was called with correct size
    sample = mock_publisher.loan_slice.return_value
    expected_message = b"1|" + pickle.dumps("test_data")
    assert mock_publisher.loan_slice.call_args[0][0] == len(expected_message)

    # Verify send was called
    assert sample.send.called


def test_producer_send_complex_data(mock_node, mock_publisher):
    """Test Producer sending complex data structures."""
    producer = Producer(service_name="test-service")

    complex_data = {"key": [1, 2, 3], "nested": {"a": "b"}}
    producer.put(complex_data, consumer_id=2)

    assert mock_publisher.loan_slice.called
    sample = mock_publisher.loan_slice.return_value
    assert sample.send.called


def test_producer_send_unpickleable(mock_node, mock_publisher):
    """Test Producer error handling for unpickleable objects."""
    producer = Producer(service_name="test-service")

    class Unpickleable:
        def __reduce__(self):
            raise pickle.PickleError("Can't pickle this!")

    with pytest.raises(pickle.PickleError):
        producer.put(Unpickleable(), consumer_id=1)


def test_producer_wait_for_subscribers(mock_node, mock_publisher):
    """Test Producer.wait_for_subscribers() method."""
    producer = Producer(service_name="test-service")

    # Test successful wait
    assert producer.wait_for_subscribers(timeout=0.1)
    assert mock_publisher.loan_slice.called


def test_producer_close(mock_node, mock_publisher):
    """Test Producer.close() method."""
    producer = Producer(service_name="test-service")
    producer.close()

    # Resources should be cleaned up
    # (In actual implementation, del is called which may be hard to test)


@pytest.mark.asyncio
async def test_async_consumer_init(mock_node, mock_subscriber):
    """Test AsyncConsumer initialization."""
    consumer = AsyncConsumer(service_name="test-service", consumer_id=1)

    assert consumer.service_name == "test-service"
    assert consumer.consumer_id == 1
    assert mock_node.service_builder.called


@pytest.mark.asyncio
async def test_async_consumer_receive(mock_node, mock_subscriber):
    """Test AsyncConsumer.receive() method."""
    test_data = {"test": "data"}
    message = b"1|" + pickle.dumps(test_data)

    # Update mock to return correct data
    sample = Mock()
    sample.payload = message
    mock_subscriber.receive.return_value = sample

    consumer = AsyncConsumer(service_name="test-service", consumer_id=1)
    received = await consumer.get()

    assert received == test_data
    assert mock_subscriber.receive.called


@pytest.mark.asyncio
async def test_async_consumer_receive_with_timeout(mock_node, mock_subscriber):
    """Test AsyncConsumer.receive() with timeout."""
    test_data = {"test": "data"}
    message = b"1|" + pickle.dumps(test_data)

    sample = Mock()
    sample.payload = message
    mock_subscriber.receive.return_value = sample

    consumer = AsyncConsumer(service_name="test-service", consumer_id=1)
    received = await consumer.get(timeout=1.0)

    assert received == test_data


@pytest.mark.asyncio
async def test_async_consumer_timeout(mock_node):
    """Test AsyncConsumer timeout handling."""
    subscriber = Mock()
    mock_node.service_builder.return_value.publish_subscribe.return_value.open_or_create.return_value.subscriber_builder.return_value.create.return_value = subscriber

    consumer = AsyncConsumer(service_name="test-service", consumer_id=1)

    # Simulate timeout by raising in executor
    with patch.object(consumer, "_sync_receive", side_effect=asyncio.TimeoutError()):
        with pytest.raises(Empty):
            await consumer.get(timeout=0.1)


@pytest.mark.asyncio
async def test_async_consumer_message_filtering(mock_node, mock_subscriber):
    """Test that AsyncConsumer filters messages by consumer_id."""
    # Message for consumer 2, not 1
    message = b"2|" + pickle.dumps({"test": "data"})

    sample = Mock()
    sample.payload = message
    mock_subscriber.receive.return_value = sample

    consumer = AsyncConsumer(service_name="test-service", consumer_id=1)

    with pytest.raises(Empty):
        await consumer.get()


@pytest.mark.asyncio
async def test_async_consumer_invalid_message_format(mock_node, mock_subscriber):
    """Test AsyncConsumer handles invalid message format."""
    # Message without consumer_id prefix
    message = pickle.dumps({"test": "data"})

    sample = Mock()
    sample.payload = message
    mock_subscriber.receive.return_value = sample

    consumer = AsyncConsumer(service_name="test-service", consumer_id=1)

    with pytest.raises(ValueError):
        await consumer.get()


def test_async_consumer_close(mock_node):
    """Test AsyncConsumer.close() method."""
    subscriber = Mock()
    mock_node.service_builder.return_value.publish_subscribe.return_value.open_or_create.return_value.subscriber_builder.return_value.create.return_value = subscriber

    consumer = AsyncConsumer(service_name="test-service", consumer_id=1)
    consumer.close()

    # Resources should be cleaned up
    # (In actual implementation, del is called which may be hard to test)
