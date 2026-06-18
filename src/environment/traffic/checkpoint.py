from collections.abc import Callable
from src.environment.traffic.vehicle import Vehicle


class Checkpoint:
    def __init__(self, position, callback: Callable[[Vehicle], None]):
        self.position = position
        self.callback = callback

    def acquire(self, vehicle):
        self.callback(vehicle)
