from src.environment.traffic.traffic_light import TrafficLight
from src.game.constants import TRAFFIC_LIGHT_POSITIONS, CELL_SIZE
from src.game.colors import TL_ON_COLOR, TL_OFF_COLOR
import pygame


class TrafficLightService:
    def __init__(self):
        self.taffic_light_count = len(TRAFFIC_LIGHT_POSITIONS)
        self.traffic_lights = [TrafficLight(index=index) for index in range(self.taffic_light_count)]

    def enable(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        traffic_light.enable()

    def disable(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        traffic_light.disable()

    def toggle(self, traffic_light_index):
        if self.is_enabled(traffic_light_index):
            self.disable(traffic_light_index)
            return

        self.enable(traffic_light_index)

    def disable_all(self):
        for traffic_light in self.traffic_lights:
            traffic_light.disable()

    def is_enabled(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        return traffic_light.is_enabled

    def state(self):
        return [traffic_light.is_enabled for traffic_light in self.traffic_lights]

    def position(self, index):
        return TRAFFIC_LIGHT_POSITIONS[index]

    def draw(self, surface: pygame.Surface):
        for index, traffic_light in enumerate(self.traffic_lights):
            tl_x, tl_y = TRAFFIC_LIGHT_POSITIONS[index]
            tl_x = tl_x * CELL_SIZE + CELL_SIZE / 2
            tl_y = tl_y * CELL_SIZE + CELL_SIZE / 2

            pygame.draw.circle(
                surface,
                (0, 0, 0),
                (tl_x, tl_y),
                12.0
            )

            pygame.draw.circle(
                surface,
                TL_ON_COLOR if traffic_light.is_enabled else TL_OFF_COLOR,
                (tl_x, tl_y),
                8.0
            )
