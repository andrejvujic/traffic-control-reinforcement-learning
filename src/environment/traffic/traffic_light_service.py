from src.environment.traffic.traffic_light import TrafficLight
from src.game.constants import COMPATIBLE_TRAFFIC_LIGHT_GROUPS, TOTAL_LANES, TRAFFIC_LIGHT_POSITIONS, CELL_SIZE
from src.game.colors import TL_ON_COLOR, TL_OFF_COLOR
import pygame


class TrafficLightService:
    def __init__(self):
        self.traffic_lights: list[TrafficLight] = [
            TrafficLight(lane_index=lane_index) for lane_index in range(TOTAL_LANES)
        ]

    def turn_green(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        traffic_light.turn_green()

    def turn_red(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        traffic_light.turn_red()

    def turn_all_red(self):
        for traffic_light in self.traffic_lights:
            traffic_light.turn_red()

    def toggle(self, traffic_light_index):
        if self.is_passable(traffic_light_index):
            self.turn_red(traffic_light_index)
            return

        self.turn_green(traffic_light_index)

    def is_passable(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        return traffic_light.is_passable()

    def state(self):
        return [traffic_light.is_passable() for traffic_light in self.traffic_lights]

    def set_state(self, traffic_light_state):
        for index, is_passable in enumerate(traffic_light_state):
            if is_passable:
                self.traffic_lights[index].turn_green()
                continue

            self.traffic_lights[index].turn_red()

    def evaluate_state(self):
        active_lanes = self.__active_lane_indices()
        if not active_lanes:
            return -5.0

        if len(active_lanes) == 1:
            return -1.0

        if not self.__are_active_lanes_compatible(active_lanes):
            return -30.0

        return 5.0 * len(active_lanes)

    def __active_lane_indices(self):
        return {traffic_light.lane_index for traffic_light in self.traffic_lights if traffic_light.is_passable()}

    def __are_active_lanes_compatible(self, active_lanes):
        return any(
            active_lanes.issubset(compatible_lanes)
            for compatible_lanes in COMPATIBLE_TRAFFIC_LIGHT_GROUPS
        )

    def reset(self):
        self.previous_action = None
        self.repeated_action = False
        self.turn_all_red()

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
                TL_ON_COLOR if traffic_light.is_passable() else TL_OFF_COLOR,
                (tl_x, tl_y),
                8.0
            )
