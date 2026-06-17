from src.game.constants import CELL_SIZE, ROWS, COLUMNS

import pygame


class Car:
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

    def draw(self, surface: pygame.Surface):
        surface.blit(
            pygame.transform.rotate(
                self.texture,
                self.angle
            ),
            (self.x * CELL_SIZE, self.y * CELL_SIZE)
        )

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

    def is_alive(self):
        return 0 <= self.x < COLUMNS and 0 <= self.y < ROWS
