import pygame
import math

from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE, TRAFFIC_LIGHT_POSITIONS
from src.game.constants import TRAFFIC_LIGHT_PHASES
from src.game.map import Map

from src.environment.traffic.traffic_light_service import TrafficLightService
from src.environment.traffic.vehicle_service import VehicleService

from src.environment.agents.agent import Agent
from src.environment.agents.random.random_agent import RandomAgent
from src.environment.agents.basic.basic_agent import BasicAgent
from src.environment.agents.dqn.dqn_agent import DQNAgent
from src.environment.agents.ppo.ppo_agent import PPOAgent


class Game:
    def __init__(self):
        pygame.init()

        pygame.display.set_caption('Traffic Simulation')
        self.surface = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(name='Consolas', size=24)

        self.traffic_light_service = TrafficLightService()
        self.vechicle_service = VehicleService(traffic_light_service=self.traffic_light_service)
        self.map_ = Map()

        random_agent = RandomAgent()
        basic_agent = BasicAgent()

        dqn_agent = DQNAgent()
        dqn_agent.load('training_output/dqn/1782752102/model.pt')

        ppo_agent = PPOAgent()
        ppo_agent.load('training_output/ppo/1782736882/model.pt')

        self.agents: list[Agent] = [
            random_agent,
            basic_agent,
            dqn_agent,
            ppo_agent
        ]

        self.selected_agent_index = 0

    def run(self):
        running = True

        current_state = self.vechicle_service.state()
        episode_length = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = self.__next_action(current_state)
            if action < len(TRAFFIC_LIGHT_PHASES):
                self.traffic_light_service.apply_phase(action)
            else:
                self.traffic_light_service.turn_all_red()

            new_state, _, terminated, truncated = self.vechicle_service.update()

            episode_length = episode_length + 1

            if terminated or truncated:
                self.vechicle_service.reset()
                self.traffic_light_service.reset()

                new_state = self.vechicle_service.state()
                episode_length = 0

            current_state = new_state

            self.__render()

    def __next_action(self, current_state):
        selected_agent = self.agents[self.selected_agent_index]
        return selected_agent.next_action(current_state)

    def __render(self):
        self.surface.fill((0, 0, 0))

        self.map_.draw(self.surface)
        self.traffic_light_service.draw(self.surface)
        self.vechicle_service.draw(self.surface)

        pygame.display.flip()
        self.clock.tick(60)

    def render_hud(self):
        text_surface = self.font.render(
            f'Cars Passed -> {self.vechicle_service.total_cars_passed:5d}',
            True,
            (255, 255, 255), (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 0))

        text_surface = self.font.render(
            f'Average Ticks Waiting (Cars) -> {self.vechicle_service.average_ticks_waiting_cars():.3f} ticks',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 24))

        text_surface = self.font.render(
            f'Trains Passed -> {self.vechicle_service.total_trains_passed:5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 48))

        text_surface = self.font.render(
            f'Average Ticks Waiting (Trains) -> {self.vechicle_service.average_ticks_waiting_trains():.3f} ticks',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 72))

        text_surface = self.font.render(
            f'Total Vehicles -> {self.vechicle_service.active_vehicles():5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 96))

        text_surface = self.font.render(
            f'Episode Length -> {self.vechicle_service.ticks:5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 120))
