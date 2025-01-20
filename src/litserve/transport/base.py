from abc import ABC, abstractmethod
from typing import Any, Optional

# TODO: raise NotImplemented error for all methods


class MessageTransport(ABC):
    @abstractmethod
    def send(self, item: Any, consumer_id: int) -> None:
        """Send a message to a consumer in the main process."""
        pass

    @abstractmethod
    async def areceive(self, timeout: Optional[int] = None, consumer_id: Optional[int] = None) -> dict:
        """Receive a message from model workers or any publisher."""
        pass

    def close(self) -> None:
        """Clean up resources if needed (e.g., sockets, processes)."""
        pass
