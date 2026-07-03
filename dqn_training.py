from src.environment.traffic.vehicle_service import VehicleService
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.map import Map
from src.environment.agents.dqn.dqn_agent import DQNAgent
from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from src.game.constants import TRAFFIC_LIGHT_PHASES
from src.game.utilities import append_training_history, log_training_message, render_debug_frame
from src.game.utilities import save_agent_training_checkpoint

import pygame
import time
import os

pygame.display.init()
surface = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)
clock = pygame.time.Clock()

traffic_light_service = TrafficLightService()
vehicle_service = VehicleService(
    traffic_light_service=traffic_light_service
)
map_ = Map()

training_start_time = time.time()
output_path = os.path.join(
    'training_output', 'dqn', f'{training_start_time:.0f}'
)
model_path = os.path.join(output_path, 'model.pt')

pma_history_output_path = os.path.join(output_path, 'pma')
raw_history_output_path = os.path.join(output_path, 'raw')

os.makedirs(raw_history_output_path, exist_ok=True)
os.makedirs(pma_history_output_path, exist_ok=True)

HEADLESS = True
CHECKPOINT_INTERVAL = 50000
MOVING_AVERAGE_WINDOW_SIZE = 50

TARGET_TICKS = 2000000
EXPLORATION_FRACTION = 0.6
START_EPSILON = 1.0
END_EPSILON = 0.05

UPDATE_INTERVAL = 1
SYNC_INTERVAL = 2000
LEARNING_STARTS = 5000

total_reward_history = []
epsilon_history = []
tick_count_history = []
cars_waiting_ticks_history = []
trains_waiting_ticks_history = []
total_cars_passed_history = []
total_trains_passed_history = []


def calculate_epsilon(tick_number):
    exploration_ticks = TARGET_TICKS * EXPLORATION_FRACTION
    progress = min(tick_number / exploration_ticks, 1.0)
    return START_EPSILON - progress * (START_EPSILON - END_EPSILON)


def save_training_checkpoint():
    save_agent_training_checkpoint(
        agent,
        model_path,
        pma_history_output_path,
        MOVING_AVERAGE_WINDOW_SIZE,
        total_reward_history,
        tick_count_history,
        cars_waiting_ticks_history,
        trains_waiting_ticks_history,
        total_cars_passed_history,
        total_trains_passed_history,
        raw_history_output_path
    )


def save_history():
    append_training_history(
        vehicle_service,
        game_reward,
        game_ticks,
        cars_waiting_ticks_history,
        trains_waiting_ticks_history,
        total_cars_passed_history,
        total_trains_passed_history,
        total_reward_history,
        tick_count_history
    )


agent = DQNAgent()
training_ticks = 0
game_ticks = 0
ticks_since_sync = 0
ticks_since_update = 0
next_checkpoint_tick = CHECKPOINT_INTERVAL

game_reward = 0.0
game_index = 0


while training_ticks < TARGET_TICKS:
    game_index = game_index + 1
    game_ticks = 0
    game_reward = 0.0

    terminated_flag, truncated_flag = False, False

    traffic_light_service.reset()
    vehicle_service.reset()
    current_state = vehicle_service.state()

    epsilon = calculate_epsilon(training_ticks)

    while not terminated_flag and not truncated_flag and training_ticks < TARGET_TICKS:
        epsilon = calculate_epsilon(training_ticks)
        action = agent.next_action(
            current_state,
            epsilon=epsilon
        )

        if action < len(TRAFFIC_LIGHT_PHASES):
            traffic_light_service.apply_phase(action)
        else:
            traffic_light_service.turn_all_red()

        new_state, reward, terminated_flag, truncated_flag = vehicle_service.update()
        game_reward = game_reward + reward

        if not HEADLESS:
            render_debug_frame(
                surface,
                map_,
                traffic_light_service,
                vehicle_service,
                clock
            )

        agent.remember(
            current_state,
            action,
            new_state,
            reward,
            terminated_flag
        )

        current_state = new_state

        game_ticks = game_ticks + 1
        training_ticks = training_ticks + 1
        ticks_since_sync = ticks_since_sync + 1
        ticks_since_update = ticks_since_update + 1

        if training_ticks >= LEARNING_STARTS:
            if ticks_since_update >= UPDATE_INTERVAL:
                ticks_since_update = 0
                agent.learn()

        if ticks_since_sync >= SYNC_INTERVAL:
            ticks_since_sync = 0
            agent.sync_networks()

    save_history()

    if game_index == 1 or training_ticks >= next_checkpoint_tick:
        log_training_message(output_path, f'Game Done -> {game_index:5d} | Ticks -> {training_ticks:8d} / {TARGET_TICKS} | Training Progress -> {training_ticks / TARGET_TICKS * 100.0:.2f}% | Epsilon -> {epsilon:.3f}')
        save_training_checkpoint()
        next_checkpoint_tick = training_ticks + CHECKPOINT_INTERVAL

training_duration = time.time() - training_start_time
log_training_message(output_path, f'Training Done | Took -> {training_duration / 3600.0:.1f} hours | Ticks -> {training_ticks}')

save_training_checkpoint()
log_training_message(output_path, f'Model Location -> \'{model_path}\'')
