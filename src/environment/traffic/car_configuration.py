from dataclasses import dataclass, asdict
from src.environment.traffic.car_checkpoint import CarCheckpoint


@dataclass
class CarConfiguration:
    lane_index: int
    initial_direction: tuple[int, int]
    initial_angle: float
    checkpoints: list[CarCheckpoint]

    def __init__(
        self,
        lane_index,
        initial_direction,
        initial_angle,
        checkpoints
    ):
        self.lane_index = lane_index
        self.initial_direction = initial_direction
        self.initial_angle = initial_angle
        self.checkpoints = checkpoints

    def as_dict(self):
        return asdict(self)
