from src.game.constants import TRAIN_LENGTH, CELL_SIZE, ROWS, COLUMNS
from src.environment.traffic.vehicle import Vehicle

import pygame


class Train(Vehicle):
    def __init__(
        self,
        texture,
        lane_index,
        initial_position,
        initial_direction,
        initial_angle,
        checkpoints
    ):
        super().__init__(
            texture,
            lane_index,
            initial_position,
            initial_direction,
            initial_angle,
            checkpoints
        )

    def draw(self, surface: pygame.Surface):
        x, y = self.x, self.y

        if self.angle in [0.0, 180.0]:
            dx, _ = self.direction
            x = x - dx * (TRAIN_LENGTH - 1)

        surface.blit(
            pygame.transform.rotate(
                self.texture,
                self.angle
            ),
            (x * CELL_SIZE, y * CELL_SIZE)
        )

    def occupied_positions(self):
        x, y = self.x, self.y
        dx, dy = self.direction

        occupied_positions = []
        for index in range(TRAIN_LENGTH):
            occupied_positions.append(
                (x - index * dx, y - index * dy)
            )
        return occupied_positions

    def collides_with(self, other):
        if self == other:
            return False

        from src.environment.traffic.car import Car
        if isinstance(other, Car):
            return other.position() in self.occupied_positions()

        if isinstance(other, Train):
            other_positions = other.occupied_positions()
            return any(position in other_positions for position in self.occupied_positions())

        raise Exception(f'Collision checking not implemented for {type(other)}')
