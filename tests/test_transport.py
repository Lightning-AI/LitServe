import asyncio
import multiprocessing as mp
from queue import Empty
from unittest.mock import MagicMock, patch

import pytest

from litserve.transport.factory import TransportConfig, create_transport_from_config
from litserve.transport.process_transport import MPQueueTransport


class TestMPQueueTransport:
    @pytest.fixture
    def manager(self):
        manager = mp.Manager()
        yield manager
        manager.shutdown()

    @pytest.fixture
    def queues(self, manager):
        return [manager.Queue() for _ in range(2)]

    @pytest.fixture
    def transport(self, manager, queues):
        return MPQueueTransport(manager, queues)

    def test_init(self, transport, queues):
        """Test that the transport initializes correctly."""
        assert transport._queues == queues
        assert transport._closed is False

    def test_send(self, transport, queues):
        test_item = {"test": "data"}
        consumer_id = 0

        transport.send(test_item, consumer_id)

        assert queues[consumer_id].get() == test_item

    def test_send_when_closed(self, transport):
        transport._closed = True
        test_item = {"test": "data"}
        consumer_id = 0

        result = transport.send(test_item, consumer_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_areceive(self, transport, queues):
        test_item = {"test": "data"}
        consumer_id = 0
        queues[consumer_id].put(test_item)

        result = await transport.areceive(consumer_id)

        assert result == test_item

    @pytest.mark.asyncio
    async def test_areceive_when_closed(self, transport):
        transport._closed = True
        consumer_id = 0

        with pytest.raises(asyncio.CancelledError, match="Transport closed"):
            await transport.areceive(consumer_id)

    @pytest.mark.asyncio
    async def test_areceive_timeout(self, transport):
        consumer_id = 0
        timeout = 0.1

        with pytest.raises(Empty):
            await transport.areceive(consumer_id, timeout=timeout)

    @pytest.mark.asyncio
    async def test_areceive_cancellation(self, transport):
        consumer_id = 0

        with patch("asyncio.to_thread", side_effect=asyncio.CancelledError), pytest.raises(asyncio.CancelledError):
            await transport.areceive(consumer_id)

    def test_close(self, transport, queues):
        transport.close()

        assert transport._closed is True
        for queue in queues:
            assert queue.get() is None

    def test_reduce(self, transport, queues):
        cls, args = transport.__reduce__()

        assert cls == MPQueueTransport
        assert args == (None, queues)


class TestTransportFactory:
    @pytest.fixture
    def mock_manager(self):
        return MagicMock()

    def test_create_mp_transport(self, mock_manager):
        config_dict = {"transport_type": "mp", "num_consumers": 2}
        config = TransportConfig(**config_dict)

        with patch("litserve.transport.factory._create_mp_transport") as mock_create:
            mock_create.return_value = MPQueueTransport(mock_manager, [MagicMock(), MagicMock()])

            transport = create_transport_from_config(config)

            assert isinstance(transport, MPQueueTransport)
            mock_create.assert_called_once()

    def test_create_transport_invalid_type(self):
        with patch("litserve.transport.factory.TransportConfig.model_validate") as mock_validate:
            mock_validate.return_value = MagicMock(transport_type="invalid")

            with pytest.raises(ValueError, match="Invalid transport type"):
                create_transport_from_config(mock_validate.return_value)


@pytest.mark.integration
class TestTransportIntegration:
    """Integration tests for the transport system."""

    @pytest.fixture
    def mock_transport(self):
        transport = MagicMock()
        transport._closed = False
        transport._waiting_tasks = []

        transport.send = MagicMock()

        async def mock_areceive(consumer_id, timeout=None, block=True):
            current_task = asyncio.current_task()
            transport._waiting_tasks.append(current_task)

            try:
                if transport._closed:
                    raise asyncio.CancelledError("Transport closed")

                await asyncio.sleep(10)  # Long sleep to ensure we'll be cancelled

                # This should only be reached if not cancelled
                return ("test_id", {"test": "data"})
            finally:
                # Clean up task reference
                if current_task in transport._waiting_tasks:
                    transport._waiting_tasks.remove(current_task)

        transport.areceive = mock_areceive

        def mock_close():
            transport._closed = True
            for task in transport._waiting_tasks:
                task.cancel()

        transport.close = mock_close

        return transport

    @pytest.mark.asyncio
    async def test_send_receive_cycle(self, mock_transport):
        """Test a complete send-receive cycle."""
        # Arrange
        test_item = ("test_id", {"test": "data"})
        consumer_id = 0

        # Act - Send
        mock_transport.send(test_item, consumer_id)

        # Act - Receive
        result = await mock_transport.areceive(consumer_id)

        # Assert
        assert result == test_item
        mock_transport.send.assert_called_once_with(test_item, consumer_id)

    @pytest.mark.asyncio
    async def test_shutdown_sequence(self, mock_transport):
        """Test the shutdown sequence works correctly."""
        # Arrange
        consumer_id = 0

        async def receive_task():
            try:
                await mock_transport.areceive(consumer_id)
                return False  # Should not reach here if cancelled
            except asyncio.CancelledError:
                return True  # Successfully cancelled

        task = asyncio.create_task(receive_task())
        await asyncio.sleep(0.1)

        # Act
        mock_transport.close()
        result = await asyncio.wait_for(task, timeout=2.0)

        # Assert
        assert result is True
