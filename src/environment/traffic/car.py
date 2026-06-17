from src.game.constants import CELL_SIZE, ROWS, COLUMNS
from src.environment.traffic.vehicle import Vehicle
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

    def is_alive(self):
        return 0 <= self.x < COLUMNS and 0 <= self.y < ROWS

    def collides_with(self, other):
        if isinstance(other, Car):
            return other.lane_index != self.lane_index and other.position() == self.position()

        raise Exception(f'Collision checking not implemented for {type(other)}')
