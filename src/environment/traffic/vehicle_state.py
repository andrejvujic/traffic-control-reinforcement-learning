from enum import Enum


class VehicleState(Enum):
    PASSED_INTERSECTION = 0
    IN_INTERSECTION = 1
    APPROACHING_INTERSECTION = 2
