from src.game.constants import CELL_SIZE, ROWS, COLUMNS
from src.environment.traffic.vehicles.vehicle import Vehicle

import pygame


class Car(Vehicle):
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
        surface.blit(
            pygame.transform.rotate(
                self.texture,
                self.angle
            ),
            (self.x * CELL_SIZE, self.y * CELL_SIZE)
        )

    def collides_with(self, other):
        if self == other or (isinstance(other, Vehicle) and self.lane_index == other.lane_index):
            return False

        if isinstance(other, Car):
            return other.lane_index != self.lane_index and other.position() == self.position()

        from src.environment.traffic.vehicles.trains.train import Train
        if isinstance(other, Train):
            return self.position() in other.occupied_positions()

        raise Exception(f'Collision checking not implemented for {type(other)}')
