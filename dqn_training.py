from src.environment.traffic.vehicle_service import VehicleService
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.map import Map
from src.environment.agents.dqn.dqn_agent import DQNAgent
from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from src.game.constants import TRAFFIC_LIGHT_PHASES

import matplotlib.pyplot as plt
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
map = Map()

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


def calculate_partial_moving_averages(values, window_size):
    moving_averages = []
    for index in range(len(values)):
        if index + 1 < window_size:
            moving_averages.append(
                sum(values[0: index + 1]) / (index + 1)
            )
            continue

        moving_averages.append(
            sum(values[index - window_size + 1: index + 1]) / window_size
        )

    return moving_averages


def save_pma_history_plot(file_name, value_history, value_label, plot_title):
    figure = plt.figure(figsize=(12, 6))

    moving_averages = calculate_partial_moving_averages(
        value_history,
        MOVING_AVERAGE_WINDOW_SIZE
    )

    games = list(
        range(1, len(value_history) + 1)
    )

    plt.plot(
        games,
        moving_averages
    )

    plt.xlabel('Game')
    plt.ylabel(value_label)
    plt.title(plot_title)

    figure.savefig(
        os.path.join(pma_history_output_path, file_name),
    )
    plt.close(figure)


def save_raw_history_plot(file_name, value_history, value_label, plot_title):
    figure = plt.figure(figsize=(12, 6))

    games = list(
        range(1, len(value_history) + 1)
    )

    plt.plot(
        games,
        value_history
    )

    plt.xlabel('Game')
    plt.ylabel(value_label)
    plt.title(plot_title)

    figure.savefig(
        os.path.join(raw_history_output_path, file_name),
    )
    plt.close(figure)


def save_training_checkpoint():
    os.makedirs(
        output_path,
        exist_ok=True
    )

    agent.save(model_path)

    save_raw_history_plot(
        'total_reward_history.jpg',
        total_reward_history,
        'Total Reward',
        f'Total Reward History'
    )

    save_raw_history_plot(
        'game_length_history.jpg',
        tick_count_history,
        'Game Length (Ticks)',
        f'Game Length History'
    )

    save_raw_history_plot(
        'car_ticks_waiting_history.jpg',
        cars_waiting_ticks_history,
        'Average Waiting Time (Cars)',
        f'Average Ticks Waiting History'
    )

    save_raw_history_plot(
        'train_ticks_waiting_history.jpg',
        trains_waiting_ticks_history,
        'Average Ticks Waiting',
        f'Average Ticks Waiting History (Trains)'
    )

    save_raw_history_plot(
        'total_cars_passed_history.jpg',
        total_cars_passed_history,
        'Cars Passed',
        f'Cars Passed History',
    )

    save_raw_history_plot(
        'total_trains_passed_history.jpg',
        total_trains_passed_history,
        'Trains Passed',
        f'Trains Passed History'
    )

    save_pma_history_plot(
        'total_reward_history.jpg',
        total_reward_history,
        'Total Reward',
        f'Total Reward History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_pma_history_plot(
        'game_length_history.jpg',
        tick_count_history,
        'Game Length (Ticks)',
        f'Game Length History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_pma_history_plot(
        'car_ticks_waiting_history.jpg',
        cars_waiting_ticks_history,
        'Average Ticks Waiting',
        f'Average Ticks Waiting History (Cars) ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_pma_history_plot(
        'train_ticks_waiting_history.jpg',
        trains_waiting_ticks_history,
        'Average Ticks Waiting',
        f'Average Ticks Waiting History (Trains) ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_pma_history_plot(
        'total_cars_passed_history.jpg',
        total_cars_passed_history,
        'Cars Passed',
        f'Cars Passed History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)',
    )

    save_pma_history_plot(
        'total_trains_passed_history.jpg',
        total_trains_passed_history,
        'Trains Passed',
        f'Trains Passed History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )


def training_log(message):
    print(message)
    with open(os.path.join(output_path, 'training_log.txt'), 'a') as f:
        f.write(f'{message}\n')


def save_history():
    cars_waiting_ticks_history.append(
        vehicle_service.average_ticks_waiting_cars()
    )
    trains_waiting_ticks_history.append(
        vehicle_service.average_ticks_waiting_trains()
    )
    total_cars_passed_history.append(vehicle_service.total_cars_passed)
    total_trains_passed_history.append(vehicle_service.total_trains_passed)
    total_reward_history.append(game_reward)
    tick_count_history.append(game_ticks)


def debug_render():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit

        surface.fill('black')

        map.draw(surface)
        traffic_light_service.draw(surface)
        vehicle_service.draw(surface)

        pygame.display.flip()
        clock.tick()


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
            debug_render()

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
        training_log(f'Game Done -> {game_index:5d} | Ticks -> {training_ticks:8d} / {TARGET_TICKS} | Training Progress -> {training_ticks / TARGET_TICKS * 100.0:.2f}% | Epsilon -> {epsilon:.3f}')
        save_training_checkpoint()
        next_checkpoint_tick = training_ticks + CHECKPOINT_INTERVAL

training_duration = time.time() - training_start_time
training_log(f'Training Done | Took -> {training_duration / 3600.0:.1f} hours | Ticks -> {training_ticks}')

save_training_checkpoint()
training_log(f'Model Location -> \'{model_path}\'')
