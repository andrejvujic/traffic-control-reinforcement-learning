from abc import ABC, abstractmethod
from src.game.constants import ROWS, COLUMNS
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

    def move(self):
        dx, dy = self.direction
        self.x = self.x + dx
        self.y = self.y + dy

        for checkpoint in self.checkpoints:
            checkpoint(self)

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
