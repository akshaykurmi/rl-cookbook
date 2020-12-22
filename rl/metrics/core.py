from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, buffer_size, name):
        self.buffer_size = buffer_size
        self.name = name

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def record(self, transition):
        pass

    @abstractmethod
    def compute(self):
        pass
