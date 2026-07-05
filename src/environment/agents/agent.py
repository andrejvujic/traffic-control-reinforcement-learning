from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def next_action(self, *args, **kwargs) -> int:
        pass
