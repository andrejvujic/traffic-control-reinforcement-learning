import pygame
import math

from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE, TRAFFIC_LIGHT_POSITIONS
from src.game.map import Map

from src.environment.traffic.traffic_light_service import TrafficLightService
from src.environment.traffic.vehicle_service import VehicleService


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Traffic Simulation')
        self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.traffic_light_service = TrafficLightService()
        self.vechicle_service = VehicleService(traffic_light_service=self.traffic_light_service)
        self.map = Map()

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_traffic_light_click(event.pos)

            self.surface.fill((0, 0, 0))

            self.map.draw(self.surface)
            self.traffic_light_service.draw(self.surface)
            self.vechicle_service.draw(self.surface)

            pygame.display.flip()

            self.clock.tick(60)

            self.vechicle_service.update()

        pygame.quit()

    def handle_traffic_light_click(self, click_position):
        click_x, click_y = click_position

        tl_distances = []
        for tl_position in TRAFFIC_LIGHT_POSITIONS:
            tl_x, tl_y = tl_position
            tl_x = tl_x * CELL_SIZE + CELL_SIZE / 2
            tl_y = tl_y * CELL_SIZE + CELL_SIZE / 2

            tl_distances.append(
                math.sqrt(
                    (tl_x - click_x) ** 2 + (tl_y - click_y) ** 2
                )
            )

        tl_index = min(
            range(
                len(TRAFFIC_LIGHT_POSITIONS)
            ),
            key=lambda index: tl_distances[index]
        )

        if tl_distances[tl_index] < 12.0:
            self.traffic_light_service.toggle(tl_index)
