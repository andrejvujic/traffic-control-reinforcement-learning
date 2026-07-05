import pygame
import math

from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, CANVAS_WIDTH
from src.game.constants import CELL_SIZE, TRAFFIC_LIGHT_POSITIONS
from src.game.constants import TRAFFIC_LIGHT_PHASES
from src.game.map import Map

from src.environment.traffic.traffic_lights.traffic_light_service import TrafficLightService
from src.environment.traffic.vehicles.vehicle_service import VehicleService

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
        self.arrow_left_icon = pygame.transform.scale(
            pygame.image.load('assets/icons/arrow_left.bmp').convert_alpha(),
            (32, 32)
        )
        self.arrow_right_icon = pygame.transform.scale(
            pygame.image.load('assets/icons/arrow_right.bmp').convert_alpha(),
            (32, 32)
        )
        self.minus_icon = pygame.transform.scale(
            pygame.image.load('assets/icons/minus.bmp').convert_alpha(),
            (32, 32)
        )
        self.plus_icon = pygame.transform.scale(
            pygame.image.load('assets/icons/plus.bmp').convert_alpha(),
            (32, 32)
        )
        self.play_icon = pygame.transform.scale(
            pygame.image.load('assets/icons/play.bmp').convert_alpha(),
            (32, 32)
        )
        self.pause_icon = pygame.transform.scale(
            pygame.image.load('assets/icons/pause.bmp').convert_alpha(),
            (32, 32)
        )
        self.restart_icon = pygame.transform.scale(
            pygame.image.load('assets/icons/restart.bmp').convert_alpha(),
            (32, 32)
        )
        self.arrow_left_rect = self.arrow_left_icon.get_rect()
        self.arrow_right_rect = self.arrow_right_icon.get_rect()
        self.minus_rect = self.minus_icon.get_rect()
        self.plus_rect = self.plus_icon.get_rect()
        self.play_rect = self.play_icon.get_rect()
        self.pause_rect = self.pause_icon.get_rect()
        self.restart_rect = self.restart_icon.get_rect()

        self.speed_options = [5, 15, 25, 50, 100, 500, 0]
        self.selected_speed_index = 0
        self.paused = False

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
        self.collision_count = 0
        self.episode_length_history = []
        self.max_episode_length = 0

    def run(self):
        running = True

        current_state = self.vechicle_service.state()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.__handle_pygame_event(event):
                    current_state = self.__reset_simulation()
                    self.__reset_statistics()

            if not running:
                break

            if self.paused:
                self.__render()
                continue

            action = self.__next_action(current_state)
            if action < len(TRAFFIC_LIGHT_PHASES):
                self.traffic_light_service.apply_phase(action)
            else:
                self.traffic_light_service.turn_all_red()

            new_state, _, terminated, __class__ = self.vechicle_service.update()

            if terminated:
                self.__record_completed_episode()

                self.vechicle_service.reset()
                self.traffic_light_service.reset()
                self.collision_count = self.collision_count + 1

                new_state = self.vechicle_service.state()

            current_state = new_state

            self.__render()

    def __handle_pygame_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.__select_previous_agent()
                return True

            if event.key == pygame.K_RIGHT:
                self.__select_next_agent()
                return True

            if event.key in [pygame.K_MINUS, pygame.K_KP_MINUS]:
                self.__decrease_speed()
                return False

            if event.key in [pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS]:
                self.__increase_speed()
                return False

            if event.key == pygame.K_SPACE:
                self.__toggle_pause()
                return False

            if event.key == pygame.K_r:
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.arrow_left_rect.collidepoint(event.pos):
                self.__select_previous_agent()
                return True

            if self.arrow_right_rect.collidepoint(event.pos):
                self.__select_next_agent()
                return True

            if self.minus_rect.collidepoint(event.pos):
                self.__decrease_speed()
                return False

            if self.plus_rect.collidepoint(event.pos):
                self.__increase_speed()
                return False

            if self.play_rect.collidepoint(event.pos):
                self.__play()
                return False

            if self.pause_rect.collidepoint(event.pos):
                self.__pause()
                return False

            if self.restart_rect.collidepoint(event.pos):
                return True

        return False

    def __select_previous_agent(self):
        self.selected_agent_index = (self.selected_agent_index - 1) % len(self.agents)

    def __select_next_agent(self):
        self.selected_agent_index = (self.selected_agent_index + 1) % len(self.agents)

    def __decrease_speed(self):
        self.selected_speed_index = max(
            0,
            self.selected_speed_index - 1
        )

    def __increase_speed(self):
        self.selected_speed_index = min(
            len(self.speed_options) - 1,
            self.selected_speed_index + 1
        )

    def __play(self):
        self.paused = False

    def __pause(self):
        self.paused = True

    def __toggle_pause(self):
        self.paused = not self.paused

    def __reset_simulation(self):
        self.vechicle_service.reset()
        self.traffic_light_service.reset()
        return self.vechicle_service.state()

    def __reset_statistics(self):
        self.collision_count = 0
        self.episode_length_history = []
        self.max_episode_length = 0

    def __record_completed_episode(self):
        episode_length = self.vechicle_service.ticks
        self.episode_length_history.append(episode_length)
        self.max_episode_length = max(
            self.max_episode_length,
            episode_length
        )

    def __average_episode_length(self):
        episode_lengths = [
            *self.episode_length_history,
            self.vechicle_service.ticks
        ]

        if not episode_lengths:
            return 0.0

        return sum(episode_lengths) / len(episode_lengths)

    def __max_episode_length(self):
        return max(
            self.max_episode_length,
            self.vechicle_service.ticks
        )

    def __render(self):
        self.surface.fill((0, 0, 0))

        self.surface.set_clip(
            pygame.Rect(0, 0, CANVAS_WIDTH, SCREEN_HEIGHT)
        )

        self.map_.draw(self.surface)
        self.traffic_light_service.draw(self.surface)
        self.vechicle_service.draw(self.surface)
        self.surface.set_clip(None)

        self.__render_hud()

        pygame.display.flip()
        self.clock.tick(
            self.__selected_speed()
        )

    def __next_action(self, current_state):
        selected_agent = self.agents[self.selected_agent_index]
        return selected_agent.next_action(current_state)

    def __selected_agent_name(self):
        selected_agent = self.agents[self.selected_agent_index]
        return selected_agent.name

    def __selected_speed(self):
        return self.speed_options[self.selected_speed_index]

    def __render_hud(self):
        hud_x = CANVAS_WIDTH + 24
        hud_right = SCREEN_WIDTH - 24
        hud_y = 24
        line_height = 32

        agent_name_surface = self.font.render(
            self.__selected_agent_name(),
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        agent_center_y = hud_y + self.arrow_left_icon.get_height() / 2
        hud_center_x = CANVAS_WIDTH + (SCREEN_WIDTH - CANVAS_WIDTH) / 2
        agent_name_rect = agent_name_surface.get_rect(
            center=(hud_center_x, agent_center_y)
        )

        self.arrow_left_rect.midleft = (
            hud_x,
            agent_center_y
        )
        self.arrow_right_rect.midright = (
            hud_right,
            agent_center_y
        )

        self.surface.blit(self.arrow_left_icon, self.arrow_left_rect)
        self.surface.blit(agent_name_surface, agent_name_rect)
        self.surface.blit(self.arrow_right_icon, self.arrow_right_rect)

        selected_speed = self.__selected_speed()
        speed_surface = self.font.render(
            f'Speed: {selected_speed} ticks/s' if selected_speed else f'Speed: Max. Possible',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        speed_center_y = hud_y + line_height * 2 + self.minus_icon.get_height() / 2
        speed_rect = speed_surface.get_rect(
            center=(hud_center_x, speed_center_y)
        )

        self.minus_rect.midleft = (
            hud_x,
            speed_center_y
        )
        self.plus_rect.midright = (
            hud_right,
            speed_center_y
        )

        self.surface.blit(self.minus_icon, self.minus_rect)
        self.surface.blit(speed_surface, speed_rect)
        self.surface.blit(self.plus_icon, self.plus_rect)

        control_center_y = hud_y + line_height * 4 + self.play_icon.get_height() / 2

        self.play_rect.midleft = (
            hud_x,
            control_center_y
        )
        self.pause_rect.center = (
            hud_center_x,
            control_center_y
        )
        self.restart_rect.midright = (
            hud_right,
            control_center_y
        )

        self.surface.blit(self.play_icon, self.play_rect)
        self.surface.blit(self.pause_icon, self.pause_rect)
        self.surface.blit(self.restart_icon, self.restart_rect)

        text_surface = self.font.render(
            f'Cars Passed: {self.vechicle_service.total_cars_passed:5d}',
            True,
            (255, 255, 255), (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 6))

        text_surface = self.font.render(
            f'Avg. Car Wait: {self.vechicle_service.average_ticks_waiting_cars():.3f} ticks',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 7))

        text_surface = self.font.render(
            f'Trains Passed: {self.vechicle_service.total_trains_passed:5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 8))

        text_surface = self.font.render(
            f'Avg. Train Wait: {self.vechicle_service.average_ticks_waiting_trains():.3f} ticks',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 9))

        text_surface = self.font.render(
            f'Active Vehicles: {self.vechicle_service.active_vehicles():5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 10))

        text_surface = self.font.render(
            f'Episode Ticks: {self.vechicle_service.ticks:5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 11))

        text_surface = self.font.render(
            f'Avg. Episode: {self.__average_episode_length():.1f} ticks',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 12))

        text_surface = self.font.render(
            f'Max. Episode: {self.__max_episode_length():5d} ticks',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 13))

        text_surface = self.font.render(
            f'Collisions: {self.collision_count:5d}',
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self.surface.blit(text_surface, (hud_x, hud_y + line_height * 14))
