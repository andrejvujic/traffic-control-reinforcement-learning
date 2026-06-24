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
surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

traffic_light_service = TrafficLightService()
vehicle_service = VehicleService(traffic_light_service=traffic_light_service)
map = Map()

training_start_time = time.time()
output_path = os.path.join(
    'training_output', 'dqn', f'{training_start_time:.0f}'
)
model_path = os.path.join(output_path, 'model.pt')

os.makedirs(
    output_path,
    exist_ok=True
)

TARGET_GAMES = 7000
EPSILON_DECAY_GAMES = 5000
START_EPSILON = 1.0
END_EPSILON = 0.0

SYNC_INTERVAL = 4096
LOG_INTERVAL = 100

HEADLESS = True

ticks_since_sync = 0
ticks_since_update = 0

total_reward_history = []
epsilon_history = []
tick_count_history = []
flow_rate_history = []
car_waiting_time_history = []
train_waiting_time_history = []
total_cars_passed_history = []
total_trains_passed_history = []

agent = DQNAgent(
    in_features=30,
    out_features=len(TRAFFIC_LIGHT_PHASES) + 1
)


def calculate_epsilon(game_index):
    progress = min(game_index / EPSILON_DECAY_GAMES, 1.0)
    return START_EPSILON - progress * (START_EPSILON - END_EPSILON)


def save_history_plot(file_name, history, y_label, title):
    figure = plt.figure(figsize=(12, 6))
    x_values = list(range(1, len(history) + 1))

    plt.plot(
        x_values,
        history,
    )

    plt.xlabel('Game')
    plt.ylabel(y_label)
    plt.title(title)

    figure.savefig(
        os.path.join(output_path, file_name),
    )
    plt.close(figure)


def save_training_checkpoint():
    os.makedirs(
        output_path,
        exist_ok=True
    )

    agent.save(model_path)

    save_history_plot(
        'total_reward_history.jpg',
        total_reward_history,
        'Total Reward',
        'Total Reward History'
    )

    save_history_plot(
        'game_length_history.jpg',
        tick_count_history,
        'Game Length (Ticks)',
        'Game Length History'
    )

    save_history_plot(
        'flow_rate_history.jpg',
        flow_rate_history,
        'Flow Rate (Vehicles/Tick)',
        'Flow Rate History'
    )

    save_history_plot(
        'car_waiting_time_history.jpg',
        car_waiting_time_history,
        'Average Waiting Time (Cars)',
        'Average Car Waiting Time History'
    )

    save_history_plot(
        'train_waiting_time_history.jpg',
        train_waiting_time_history,
        'Average Waiting Time (Trains)',
        'Average Train Waiting Time History'
    )

    save_history_plot(
        'total_cars_passed_history.jpg',
        total_cars_passed_history,
        'Cars Passed',
        'Cars Passed History'
    )

    save_history_plot(
        'total_trains_passed_history.jpg',
        total_trains_passed_history,
        'Trains Passed',
        'Trains Passed History'
    )

    save_history_plot(
        'epsilon_history.jpg',
        epsilon_history,
        'Epsilon',
        'Epsilon History'
    )


def training_log(message):
    print(message)
    with open(os.path.join(output_path, 'training_log.txt'), 'a') as f:
        f.write(f'{message}\n')


for game_index in range(TARGET_GAMES):
    total_reward = 0.0
    ticks = 0

    epsilon = calculate_epsilon(game_index + 1)

    terminated_flag, truncated_flag = False, False

    traffic_light_service.reset()
    vehicle_service.reset()
    current_state = vehicle_service.state()

    while not terminated_flag and not truncated_flag:
        action, random_action_taken = agent.next_action(
            current_state,
            epsilon=epsilon
        )

        if action < len(TRAFFIC_LIGHT_PHASES):
            traffic_light_service.apply_phase(action)
        else:
            traffic_light_service.turn_all_red()

        new_state, reward, terminated_flag, truncated_flag = vehicle_service.update()
        total_reward = total_reward + reward

        agent.remember(
            current_state,
            action,
            new_state,
            reward,
            terminated_flag
        )

        if not HEADLESS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

            surface.fill((0, 0, 0))
            map.draw(surface)
            traffic_light_service.draw(surface)
            vehicle_service.draw(surface)

            pygame.display.flip()
            clock.tick()

        current_state = new_state

        agent.learn()

        if ticks_since_sync >= SYNC_INTERVAL:
            ticks_since_sync = 0
            agent.sync_networks()

        ticks = ticks + 1
        ticks_since_sync = ticks_since_sync + 1
        ticks_since_update = ticks_since_update + 1

    flow_rate_history.append(vehicle_service.flow_rate())
    car_waiting_time_history.append(vehicle_service.average_waiting_time_for_cars())
    train_waiting_time_history.append(vehicle_service.average_waiting_time_for_trains())
    total_cars_passed_history.append(vehicle_service.total_cars_passed)
    total_trains_passed_history.append(vehicle_service.total_trains_passed)
    total_reward_history.append(total_reward)
    tick_count_history.append(ticks)
    epsilon_history.append(epsilon)

    if game_index == 0 or (game_index + 1) % LOG_INTERVAL == 0:
        training_log(f'Game Done -> {game_index + 1:5d} / {TARGET_GAMES} | Training Progress -> {(game_index + 1) / TARGET_GAMES * 100.0:.2f}% | Epsilon -> {epsilon:.2f}')

    save_training_checkpoint()


training_duration = time.time() - training_start_time
training_log(f'Training Done | Took ({training_duration / 3600.0:.1f} hours)')
