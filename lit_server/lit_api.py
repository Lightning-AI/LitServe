from abc import ABC, abstractmethod


class LitAPI(ABC):
    @abstractmethod
    def setup(self, devices):
        pass

    @abstractmethod
    def predict(self, x):
        pass
