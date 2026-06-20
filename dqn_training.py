from src.environment.traffic.vehicle_service import VehicleService
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.map import Map
from src.environment.agents.dqn.dqn_agent import DQNAgent
from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from src.game.constants import TRAFFIC_LIGHT_PHASES

import pygame
pygame.display.init()
surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

import matplotlib.pyplot as plt

traffic_light_service = TrafficLightService()
vehicle_service = VehicleService(traffic_light_service=traffic_light_service)
map = Map()

agent = DQNAgent(
    in_features=30,
    out_features=len(TRAFFIC_LIGHT_PHASES) + 1
)

TARGET_GAMES = 10000

START_EPSILON = 1.0
END_EPSILON = 0.05

SYNC_INTERVAL = 512
LOG_INTERVAL = 100

HEADLESS = True


def calculate_epsilon(progress):
    return START_EPSILON - progress * (START_EPSILON - END_EPSILON)


ticks_since_sync = 0
ticks_since_update = 0

reward_history = []
epsilon_history = []
game_length_history = []

max_ticks = 0


for game_index in range(TARGET_GAMES):
    non_random_actions = 0
    random_actions = 0
    total_reward = 0.0
    total_non_random_reward = 0.0

    ticks = 0

    epsilon = calculate_epsilon(
        min(
            (game_index + 1) / (2 * TARGET_GAMES / 3),
            1.0
        )
    )

    terminated_flag, truncated_flag = False, False

    vehicle_service.reset()
    traffic_light_service.reset()
    current_state = vehicle_service.state()

    while not terminated_flag and not truncated_flag:
        ticks = ticks + 1
        ticks_since_sync = ticks_since_sync + 1
        ticks_since_update = ticks_since_update + 1

        action, random_action_taken = agent.next_action(
            current_state,
            epsilon=epsilon
        )

        if random_action_taken:
            random_actions = random_actions + 1
        else:
            non_random_actions = non_random_actions + 1

        if action < len(TRAFFIC_LIGHT_PHASES):
            traffic_light_service.apply_phase(action)
        else:
            traffic_light_service.turn_all_red()

        new_state, reward, terminated_flag, truncated_flag = vehicle_service.update()
        total_reward = total_reward + reward
        total_non_random_reward = total_non_random_reward + 0.0 if random_action_taken else reward

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
            new_state,
            reward,
            terminated_flag
        )

        current_state = new_state

        agent.learn()

        if ticks_since_sync >= SYNC_INTERVAL:
            ticks_since_sync = 0
            agent.sync_networks()

    if not max_ticks or ticks > max_ticks:
        max_ticks = ticks

    if game_index == 0 or (game_index + 1) % LOG_INTERVAL == 0:
        print(f'Game Done -> {game_index + 1:5d} / {TARGET_GAMES} | Max. Game Length -> {max_ticks:5d} | | Epsilon -> {epsilon:.2f}')
        max_ticks = 0

    reward_history.append(
        total_reward / ticks
    )
    game_length_history.append(
        ticks
    )

figure = plt.figure()
plt.plot(range(TARGET_GAMES), reward_history)
plt.xlabel('Game')
plt.ylabel('Average Reward')
plt.title('Average Reward History')
figure.savefig('reward_history.jpg')
plt.close(figure)

figure = plt.figure()
plt.plot(range(TARGET_GAMES), game_length_history)
plt.xlabel('Game')
plt.ylabel('Game Length (Ticks)')
plt.title('Game Length History')
figure.savefig('game_length_history.jpg')
plt.close(figure)

agent.save()
