from src.environment.traffic.vehicle_service import VehicleService
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.map import Map
from src.environment.agents.ppo.ppo_agent import PPOAgent
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

output_path = os.path.join('training_output', 'ppo', f'{training_start_time:.0f}')
model_path = os.path.join(output_path, 'model.pt')
os.makedirs(
    output_path,
    exist_ok=True
)

HEADLESS = True
LOG_INTERVAL = 100

TARGET_GAMES = 10000
N = 512
BATCH_SIZE = 64
EPOCHS = 8

total_reward_history = []
epsilon_history = []
tick_count_history = []
flow_rate_history = []
car_waiting_time_history = []
train_waiting_time_history = []
total_cars_passed_history = []
total_trains_passed_history = []


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


def training_log(message):
    print(message)
    with open(os.path.join(output_path, 'training_log.txt'), 'a') as f:
        f.write(f'{message}\n')


agent = PPOAgent()
total_ticks = 0

for game_index in range(TARGET_GAMES):
    ticks = 0
    total_reward = 0.0

    terminated_flag, truncated_flag = False, False

    traffic_light_service.reset()
    vehicle_service.reset()
    current_state = vehicle_service.state()

    while not terminated_flag and not truncated_flag:
        action, action_log_prob = agent.next_action(current_state)
        value = agent.evaluate_state(current_state).item()

        if action < len(TRAFFIC_LIGHT_PHASES):
            traffic_light_service.apply_phase(action)
        else:
            traffic_light_service.turn_all_red()

        total_ticks += 1
        ticks += 1

        new_state, reward, terminated_flag, truncated_flag = vehicle_service.update()
        total_reward += reward

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

        agent.remember(
            current_state,
            action,
            action_log_prob,
            reward,
            value,
            terminated_flag or truncated_flag
        )

        current_state = new_state

        if len(agent.memory) >= N:
            last_value = 0.0 if (terminated_flag or truncated_flag) else agent.evaluate_state(current_state).item()
            agent.learn(EPOCHS, BATCH_SIZE, last_value)
            agent.forget()

    flow_rate_history.append(vehicle_service.flow_rate())
    car_waiting_time_history.append(vehicle_service.average_waiting_time_for_cars())
    train_waiting_time_history.append(vehicle_service.average_waiting_time_for_trains())
    total_cars_passed_history.append(vehicle_service.total_cars_passed)
    total_trains_passed_history.append(vehicle_service.total_trains_passed)
    total_reward_history.append(total_reward)
    tick_count_history.append(ticks)

    if game_index == 0 or (game_index + 1) % LOG_INTERVAL == 0:
        training_log(f'Game Done -> {game_index + 1:5d} / {TARGET_GAMES} | Training Progress -> {(game_index + 1) / TARGET_GAMES * 100.0:.2f}%')
        save_training_checkpoint()

training_duration = time.time() - training_start_time
training_log(f'Training Done | Took ({training_duration / 3600.0:.1f} hours) | Ticks: {total_ticks}')
save_training_checkpoint()
