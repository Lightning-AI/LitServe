import asyncio
import pickle
from queue import Empty
from unittest.mock import AsyncMock, Mock, patch

import pytest
import zmq

from litserve.transport.zmq_queue import AsyncConsumer, Broker, Producer


@pytest.fixture
def mock_context():
    with patch("zmq.Context") as mock_ctx:
        socket = Mock()
        mock_ctx.return_value.socket.return_value = socket
        yield mock_ctx, socket


@pytest.fixture
def mock_async_context():
    with patch("zmq.asyncio.Context") as mock_ctx:
        socket = AsyncMock()  # Use AsyncMock for async methods
        mock_ctx.return_value.socket.return_value = socket
        yield mock_ctx, socket


def test_broker_start_stop(mock_context):
    _, socket = mock_context
    broker = Broker(use_process=False)

    # Start broker
    broker.start()
    assert socket.bind.call_count == 2  # Should bind both frontend and backend

    # Stop broker
    broker.stop()
    assert socket.close.call_count == 2  # Should close both sockets


def test_broker_error_handling(mock_context):
    """Test broker handles ZMQ errors."""
    _, socket = mock_context
    socket.bind.side_effect = zmq.ZMQError("Test error")

    broker = Broker(use_process=False)
    broker.start()
    broker.stop()

    assert socket.close.called  # Should clean up even on error


def test_producer_send(mock_context):
    _, socket = mock_context
    producer = Producer(address="test_addr")

    # Test sending simple data
    producer.put("test_data", consumer_id=1)
    sent_message = socket.send.call_args[0][0]
    consumer_id, data = sent_message.split(b"|", 1)
    assert consumer_id == b"1"
    assert pickle.loads(data) == "test_data"

    # Test sending complex data
    complex_data = {"key": [1, 2, 3]}
    producer.put(complex_data, consumer_id=2)
    sent_message = socket.send.call_args[0][0]
    consumer_id, data = sent_message.split(b"|", 1)
    assert consumer_id == b"2"
    assert pickle.loads(data) == complex_data


def test_producer_error_handling(mock_context):
    _, socket = mock_context
    producer = Producer(address="test_addr")

    # Test ZMQ error
    socket.send.side_effect = zmq.ZMQError("Test error")
    with pytest.raises(zmq.ZMQError):
        producer.put("data", consumer_id=1)

    # Test unpickleable object
    class Unpickleable:
        def __reduce__(self):
            raise pickle.PickleError("Can't pickle this!")

    with pytest.raises(pickle.PickleError):
        producer.put(Unpickleable(), consumer_id=1)


def test_producer_wait_for_subscribers(mock_context):
    _, socket = mock_context
    producer = Producer(address="test_addr")

    # Test successful wait
    assert producer.wait_for_subscribers(timeout=0.1)
    assert socket.send.called

    # Test timeout
    socket.send.side_effect = zmq.ZMQError("Would block")
    assert not producer.wait_for_subscribers(timeout=0.1)


@pytest.mark.parametrize("timeout", [1.0, None])
@pytest.mark.asyncio
async def test_async_consumer(mock_async_context, timeout):
    _, socket = mock_async_context
    consumer = AsyncConsumer(consumer_id=1, address="test_addr")

    # Setup mock received data
    test_data = {"test": "data"}
    message = b"1|" + pickle.dumps(test_data)
    socket.recv.return_value = message

    # Test receiving
    received = await consumer.get(timeout=timeout)
    assert received == test_data

    # Test timeout
    socket.recv.side_effect = asyncio.TimeoutError()
    with pytest.raises(Empty):
        await consumer.get(timeout=timeout)


@pytest.mark.asyncio
async def test_async_consumer_cleanup():
    with patch("zmq.asyncio.Context") as mock_ctx:
        socket = AsyncMock()
        mock_ctx.return_value.socket.return_value = socket

        consumer = AsyncConsumer(consumer_id=1, address="test_addr")
        await consumer.aclose()

        assert socket.close.called
        assert mock_ctx.return_value.term.called


def test_producer_cleanup():
    with patch("zmq.Context") as mock_ctx:
        socket = Mock()
        mock_ctx.return_value.socket.return_value = socket

        producer = Producer(address="test_addr")
        producer.close()

        assert socket.close.called
        assert mock_ctx.return_value.term.called
