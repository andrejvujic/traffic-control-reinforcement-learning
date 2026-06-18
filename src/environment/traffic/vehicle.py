from abc import ABC, abstractmethod
from src.environment.traffic.vehicle_state import VehicleState
import pygame


class Vehicle(ABC):
    def __init__(
        self,
        texture,
        lane_index,
        initial_position,
        initial_direction,
        initial_angle,
        checkpoints
    ):
        self.texture = texture
        self.lane_index = lane_index

        self.x, self.y = initial_position
        self.direction = initial_direction
        self.angle = initial_angle

        self.checkpoints = checkpoints

        self.state = VehicleState.APPROACHING_INTERSECTION

    @abstractmethod
    def draw(self, surface: pygame.Surface):
        pass

    def position(self):
        return self.x, self.y

    def next_position(self):
        dx, dy = self.direction
        return self.x + dx, self.y + dy

    def set_direction(self, direction):
        self.direction = direction

    def set_angle(self, angle):
        self.angle = angle

    def set_state(self, state):
        self.state = state

    def move(self):
        dx, dy = self.direction
        self.x = self.x + dx
        self.y = self.y + dy

        for checkpoint in self.checkpoints:
            if self.did_reach_checkpoint(checkpoint):
                checkpoint.acquire(self)

    @abstractmethod
    def is_alive(self):
        pass

    @abstractmethod
    def collides_with(self, other):
        pass

    def collides_with_any(self, other_vehicles):
        for other_vehicle in other_vehicles:
            if self.collides_with(other_vehicle):
                return True

        return False

    def is_approaching_intersection(self):
        return self.state == VehicleState.APPROACHING_INTERSECTION

    def is_in_intersection(self):
        return self.state == VehicleState.IN_INTERSECTION

    def has_passed_intersection(self):
        return self.state == VehicleState.PASSED_INTERSECTION

    def did_reach_checkpoint(self, checkpoint):
        return self.position() == checkpoint.position
