from multiprocessing import Manager
from typing import Literal, Optional

from pydantic import BaseModel, Field

from litserve.transport.process_transport import MPQueueTransport
from litserve.transport.zmq_queue import Broker
from litserve.transport.zmq_transport import ZMQTransport

try:
    from litserve.transport.iceoryx2_transport import Iceoryx2Transport
    from litserve.transport.iceoryx2_queue import create_iceoryx2_service_names
    ICEORYX2_AVAILABLE = True
except ImportError:
    ICEORYX2_AVAILABLE = False


class TransportConfig(BaseModel):
    transport_type: Literal["mp", "zmq", "iceoryx2"] = "mp"
    num_consumers: int = Field(1, ge=1)
    manager: Optional[Manager] = None
    consumer_id: Optional[int] = None
    frontend_address: Optional[str] = None
    backend_address: Optional[str] = None


def _create_zmq_transport(config: TransportConfig):
    broker = Broker()
    broker.start()
    config.frontend_address = broker.frontend_address
    config.backend_address = broker.backend_address
    return ZMQTransport(config.frontend_address, config.backend_address)


def _create_mp_transport(config: TransportConfig):
    queues = [config.manager.Queue() for _ in range(config.num_consumers)]
    return MPQueueTransport(config.manager, queues)


def _create_iceoryx2_transport(config: TransportConfig):
    if not ICEORYX2_AVAILABLE:
        raise ImportError(
            "iceoryx2 transport requested but not installed. "
            "Install with: pip install litserve[iceoryx2]"
        )
    frontend_service, backend_service = create_iceoryx2_service_names()
    config.frontend_address = frontend_service
    config.backend_address = backend_service
    return Iceoryx2Transport(frontend_service, backend_service)


def create_transport_from_config(config: TransportConfig):
    if config.transport_type == "mp":
        return _create_mp_transport(config)
    if config.transport_type == "zmq":
        return _create_zmq_transport(config)
    if config.transport_type == "iceoryx2":
        return _create_iceoryx2_transport(config)
    raise ValueError(f"Invalid transport type: {config.transport_type}")
