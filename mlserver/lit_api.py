from abc import ABC, abstractmethod
from fastapi import Request

class LitAPI(ABC):
    @abstractmethod
    async def setup(self):
        pass

    @abstractmethod
    async def predict(self, request: Request):
        pass