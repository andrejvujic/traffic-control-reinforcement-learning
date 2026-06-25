import pygame
import math

from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE, TRAFFIC_LIGHT_POSITIONS
from src.game.constants import TRAFFIC_LIGHT_PHASES
from src.game.map import Map

from src.environment.traffic.traffic_light_service import TrafficLightService
from src.environment.traffic.vehicle_service import VehicleService
from src.environment.agents.basic.basic_agent import BasicAgent
from src.environment.agents.dqn.dqn_agent import DQNAgent
from src.environment.agents.ppo.ppo_agent import PPOAgent


class Game:
    def __init__(self):
        pygame.init()

        pygame.display.set_caption('Traffic Simulation')
        self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(name='Consolas', size=24)

        self.should_render_hud = False

        self.traffic_light_service = TrafficLightService()
        self.vechicle_service = VehicleService(traffic_light_service=self.traffic_light_service)
        self.map = Map()

        self.dqn_agent = DQNAgent(
            in_features=30,
            out_features=len(TRAFFIC_LIGHT_PHASES) + 1
        )
        # self.dqn_agent.load('training_output/dqn/1782341084/model.pt')

        self.ppo_agent = PPOAgent()
        self.ppo_agent.load('training_output/ppo/1782396035/model.pt')

    def run(self):
        running = True

        current_state = self.vechicle_service.state()
        episode_length = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_traffic_light_click(event.pos)
                    self.handle_toggle_render_hud_click(event.pos)

            self.surface.fill((0, 0, 0))

            self.map.draw(self.surface)
            self.traffic_light_service.draw(self.surface)
            self.vechicle_service.draw(self.surface)

            if self.should_render_hud:
                self.render_hud()

            pygame.display.flip()

            self.clock.tick(60)

            # action, _ = self.dqn_agent.next_action(current_state)
            action, _ = self.ppo_agent.next_action(current_state, sample=False)

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

    def handle_toggle_render_hud_click(self, click_position):
        click_x, click_y = click_position

        if click_x <= 96 and click_y <= 96:
            self.should_render_hud = not self.should_render_hud

    def render_hud(self):
        text_surface = self.font.render(
            f'Cars Passed -> {self.vechicle_service.total_cars_passed:5d}',
            True,
            (255, 255, 255), (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 0))

        text_surface = self.font.render(
            f'Average Waiting Time (Cars) -> {self.vechicle_service.average_waiting_time_for_cars():.3f} ticks',
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
            f'Average Waiting Time (Trains) -> {self.vechicle_service.average_waiting_time_for_trains():.3f} ticks',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 72))

        text_surface = self.font.render(
            f'Flow Rate -> {self.vechicle_service.flow_rate():.3f} vehicles/tick',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 96))

        text_surface = self.font.render(
            f'Total Vehicles -> {self.vechicle_service.total_vehicles():5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 120))

        text_surface = self.font.render(
            f'Episode Length -> {self.vechicle_service.ticks:5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (0, 144))
