from src.environment.traffic.traffic_light import TrafficLight
from src.game.constants import TOTAL_LANES, TRAFFIC_LIGHT_POSITIONS, CELL_SIZE
from src.game.constants import TRAFFIC_LIGHT_PHASES
from src.game.colors import TL_ON_COLOR, TL_OFF_COLOR
import pygame


class TrafficLightService:
    def __init__(self):
        self.traffic_lights: list[TrafficLight] = [
            TrafficLight(lane_index=lane_index) for lane_index in range(TOTAL_LANES)
        ]

        self.previous_state = None

    def turn_green(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        traffic_light.turn_green()

    def turn_red(self, traffic_light_index):
        traffic_light = self.traffic_lights[traffic_light_index]
        traffic_light.turn_red()

    def turn_all_red(self):
        self.previous_state = self.state()

        for traffic_light in self.traffic_lights:
            traffic_light.turn_red()

    def apply_phase(self, phase_index):
        self.previous_state = self.state()

        active_lanes = TRAFFIC_LIGHT_PHASES[phase_index]
        for traffic_light in self.traffic_lights:
            if traffic_light.lane_index in active_lanes:
                traffic_light.turn_green()
                continue

            traffic_light.turn_red()

    def previous_action_repeated(self):
        return all([
            previously_passable == now_passable for previously_passable, now_passable in zip(
                self.previous_state,
                self.state()
            )
        ])

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

    def reset(self):
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
