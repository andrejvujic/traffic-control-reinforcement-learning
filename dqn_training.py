from src.environment.traffic.vehicle_service import VehicleService
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.environment.agents.dqn.dqn_agent import DQNAgent
from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from src.game.constants import TOTAL_LANES

import pygame
pygame.display.init()
pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

import matplotlib.pyplot as plt

traffic_light_service = TrafficLightService()
vehicle_service = VehicleService(traffic_light_service=traffic_light_service)

agent = DQNAgent(
    in_features=30,
    out_features=11
)

TARGET_GAMES = 10000

START_EPSILON = 1.0
END_EPSILON = 0.1

UPDATE_INTERVAL = 256
SYNC_INTERVAL = 512
LOG_INTERVAL = 100


def calculate_epsilon(progress):
    return START_EPSILON - progress * (START_EPSILON - END_EPSILON)


ticks_since_sync = 0
ticks_since_update = 0

reward_history = []
epsilon_history = []

for game_index in range(TARGET_GAMES):
    random_actions = 0
    total_reward = 0.0
    ticks = 0

    epsilon = calculate_epsilon(
        (game_index + 1) / TARGET_GAMES
    )

    terminated_flag, truncated_flag = False, False

    vehicle_service.reset()
    traffic_light_service.reset()
    current_state = vehicle_service.state()

    while not terminated_flag and not truncated_flag:
        ticks = ticks + 1
        ticks_since_sync = ticks_since_sync + 1
        ticks_since_update = ticks_since_update + 1

        action, is_random = agent.next_action(
            current_state,
            epsilon=epsilon
        )

        if is_random:
            random_actions = random_actions + 1

        if action < TOTAL_LANES:
            traffic_light_service.toggle(action)

        new_state, reward, terminated_flag, truncated_flag = vehicle_service.update()
        total_reward = total_reward + reward

        agent.remember(
            current_state,
            action,
            new_state,
            reward,
            terminated_flag
        )

        current_state = new_state

        if ticks_since_sync >= SYNC_INTERVAL:
            ticks_since_sync = 0
            agent.sync_networks()

        if ticks_since_update >= UPDATE_INTERVAL:
            ticks_since_update = 0
            agent.learn()

    if game_index == 0 or (game_index + 1) % LOG_INTERVAL == 0:
        print(f'Game Done -> {game_index + 1:5d} / {TARGET_GAMES} | Game Length -> {ticks:5d} | Reward -> {total_reward:5.2f} | Random Actions -> {random_actions / ticks * 100.0:.0f}% | Last Action Random -> {is_random} | Epsilon -> {epsilon:.2f}')

    reward_history.append(
        total_reward / ticks
    )

figure = plt.figure()
plt.plot(range(TARGET_GAMES), reward_history)
plt.xlabel('Game')
plt.ylabel('Average Reward')
plt.title('Average Reward History')
figure.savefig('reward_history.jpg')
plt.close(figure)

agent.save()
