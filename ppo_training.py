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
vehicle_service = VehicleService(
    traffic_light_service=traffic_light_service
)
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
MOVING_AVERAGE_WINDOW_SIZE = 50

TARGET_GAMES = 10000
UPDATE_INTERVAL = 512
BATCH_SIZE = 64
EPOCHS = 8

total_reward_history = []
epsilon_history = []
tick_count_history = []
cars_waiting_ticks_history = []
trains_waiting_ticks_history = []
total_cars_passed_history = []
total_trains_passed_history = []


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


def save_history_plot(file_name, value_history, value_label, plot_title):
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
        f'Total Reward History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_history_plot(
        'game_length_history.jpg',
        tick_count_history,
        'Game Length (Ticks)',
        f'Game Length History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_history_plot(
        'car_ticks_waiting_history.jpg',
        cars_waiting_ticks_history,
        'Average Waiting Time (Cars)',
        f'Average Ticks Waiting History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_history_plot(
        'train_ticks_waiting_history.jpg',
        trains_waiting_ticks_history,
        'Average Ticks Waiting',
        f'Average Ticks Waiting History (Trains) ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)'
    )

    save_history_plot(
        'total_cars_passed_history.jpg',
        total_cars_passed_history,
        'Cars Passed',
        f'Cars Passed History ({MOVING_AVERAGE_WINDOW_SIZE}-Game PMA)',
    )

    save_history_plot(
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


agent = PPOAgent()
training_ticks = 0
game_ticks = 0
game_reward = 0.0

for game_index in range(TARGET_GAMES):
    game_ticks = 0
    game_reward = 0.0

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

        game_ticks = game_ticks + 1

        new_state, reward, terminated_flag, truncated_flag = vehicle_service.update()
        game_reward = game_reward + reward

        if not HEADLESS:
            debug_render()

        agent.remember(
            current_state,
            action,
            action_log_prob,
            reward,
            value,
            terminated_flag or truncated_flag
        )

        current_state = new_state

        if len(agent.memory) >= UPDATE_INTERVAL:
            last_value = 0.0 if (terminated_flag or truncated_flag) else agent.evaluate_state(current_state).item()
            agent.learn(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                last_value=last_value
            )
            agent.forget()

    training_ticks = training_ticks + game_ticks
    save_history()

    if game_index == 0 or (game_index + 1) % LOG_INTERVAL == 0:
        training_log(f'Game Done -> {game_index + 1:5d} / {TARGET_GAMES} | Training Progress -> {(game_index + 1) / TARGET_GAMES * 100.0:.2f}%')
        save_training_checkpoint()

training_duration = time.time() - training_start_time
training_log(f'Training Done | Took ({training_duration / 3600.0:.1f} hours) | Ticks: {training_ticks}')

save_training_checkpoint()
training_log(f'Model Location -> \'{model_path}\'')
