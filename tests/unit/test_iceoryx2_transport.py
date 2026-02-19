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
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import iceoryx2 modules only if available
try:
    from litserve.transport.iceoryx2_transport import Iceoryx2Transport
    ICEORYX2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ICEORYX2_AVAILABLE = False

# Skip all tests in this module if iceoryx2 is not available or Windows
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not ICEORYX2_AVAILABLE,
    reason="iceoryx2 not available or Windows platform"
)


@pytest.fixture
def mock_producer():
    """Create a mock Producer."""
    with patch("litserve.transport.iceoryx2_transport.Producer") as mock_cls:
        producer = Mock()
        producer.wait_for_subscribers = Mock(return_value=True)
        producer.put = Mock()
        mock_cls.return_value = producer
        yield producer


@pytest.fixture
def mock_consumer():
    """Create a mock AsyncConsumer."""
    with patch("litserve.transport.iceoryx2_transport.AsyncConsumer") as mock_cls:
        consumer = Mock()
        consumer.get = AsyncMock(return_value={"test": "data"})
        mock_cls.return_value = consumer
        yield consumer


def test_transport_init():
    """Test Iceoryx2Transport initialization."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    assert transport.frontend_service == "frontend-service"
    assert transport.backend_service == "backend-service"
    assert transport._iceoryx2 is None


def test_transport_setup_publisher(mock_producer):
    """Test Iceoryx2Transport.setup() with 'pub' operation."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    transport.setup("pub")

    assert transport._iceoryx2 is not None
    assert mock_producer.wait_for_subscribers.called


def test_transport_setup_subscriber(mock_consumer):
    """Test Iceoryx2Transport.setup() with 'sub' operation."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    transport.setup("sub", consumer_id=1)

    assert transport._iceoryx2 is not None


def test_transport_setup_invalid_operation():
    """Test Iceoryx2Transport.setup() with invalid operation."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    with pytest.raises(ValueError, match="Invalid operation"):
        transport.setup("invalid")


def test_transport_setup_subscriber_without_consumer_id():
    """Test Iceoryx2Transport.setup() as subscriber without consumer_id."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    with pytest.raises(ValueError, match="consumer_id required"):
        transport.setup("sub")


def test_transport_send(mock_producer):
    """Test Iceoryx2Transport.send() method."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    transport.send({"test": "data"}, consumer_id=1)

    assert mock_producer.put.called
    mock_producer.put.assert_called_once_with({"test": "data"}, 1)


def test_transport_send_auto_setup(mock_producer):
    """Test that send() auto-initializes producer if not setup."""
    with patch("litserve.transport.iceoryx2_transport.Producer") as mock_cls:
        producer = Mock()
        producer.wait_for_subscribers = Mock(return_value=True)
        producer.put = Mock()
        mock_cls.return_value = producer

        transport = Iceoryx2Transport("frontend-service", "backend-service")

        # Don't call setup(), send() should auto-setup
        transport.send({"test": "data"}, consumer_id=1)

        assert producer.put.called


@pytest.mark.asyncio
async def test_transport_areceive(mock_consumer):
    """Test Iceoryx2Transport.areceive() method."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    result = await transport.areceive(timeout=1.0, consumer_id=1)

    assert result == {"test": "data"}
    assert mock_consumer.get.called


@pytest.mark.asyncio
async def test_transport_areceive_auto_setup():
    """Test that areceive() auto-initializes consumer if not setup."""
    with patch("litserve.transport.iceoryx2_transport.AsyncConsumer") as mock_cls:
        consumer = Mock()
        consumer.get = AsyncMock(return_value={"test": "data"})
        mock_cls.return_value = consumer

        transport = Iceoryx2Transport("frontend-service", "backend-service")

        # Don't call setup(), areceive() should auto-setup
        result = await transport.areceive(timeout=1.0, consumer_id=1)

        assert result == {"test": "data"}
        assert consumer.get.called


@pytest.mark.asyncio
async def test_transport_areceive_without_consumer_id():
    """Test areceive() without consumer_id before initialization."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    with pytest.raises(ValueError, match="consumer_id required"):
        await transport.areceive(timeout=1.0)


def test_transport_close(mock_producer):
    """Test Iceoryx2Transport.close() method."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")
    transport.setup("pub")

    transport.close()

    assert mock_producer.close.called


def test_transport_close_not_initialized():
    """Test close() when transport is not initialized."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    with pytest.raises(ValueError, match="Iceoryx2 not initialized"):
        transport.close()


def test_transport_pickle():
    """Test Iceoryx2Transport pickling support."""
    transport = Iceoryx2Transport("frontend-service", "backend-service")

    # Test __reduce__ returns correct tuple
    reduced = transport.__reduce__()

    assert reduced[0] == Iceoryx2Transport
    assert reduced[1] == ("frontend-service", "backend-service")
