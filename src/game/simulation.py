import pygame
from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from src.game.map import Map

from src.environment.traffic.traffic_light_service import TrafficLightService


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Traffic Simulation')
        self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.traffic_light_service = TrafficLightService()

        self.map = Map()

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.surface.fill((0, 0, 0))
            self.map.draw(self.surface)
            self.traffic_light_service.draw(self.surface)

            pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()
