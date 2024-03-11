from abc import ABC, abstractmethod


class AbstractPipeline(ABC):
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
