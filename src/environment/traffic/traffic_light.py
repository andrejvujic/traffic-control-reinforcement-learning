from enum import Enum


class TrafficLightState(Enum):
    GREEN = 0
    RED = 1


class TrafficLight:
    def __init__(self, index):
        self.index = index
        self.state = TrafficLightState.RED

    def turn_green(self):
        self.state = TrafficLightState.GREEN

    def turn_red(self):
        self.state = TrafficLightState.RED

    def is_passable(self):
        return self.state == TrafficLightState.GREEN
