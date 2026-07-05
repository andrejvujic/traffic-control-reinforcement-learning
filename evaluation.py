from src.environment.agents.ppo.ppo_evaluator import PPOAgentEvaluator
from src.environment.agents.dqn.dqn_evaluator import DQNAgentEvaluator
from src.environment.agents.basic.basic_evaluator import BasicAgentEvaluator
from src.environment.agents.random.random_evaluator import RandomAgentEvaluator
from src.environment.agents.agent_evaluator import AgentEvaluator
from src.game.constants import LANE_NAMES

import matplotlib.pyplot as plt
import numpy as np
import os
import time

start_time = time.time()
output_path = os.path.join('evaluation_output', f'{start_time:.0f}')
os.makedirs(
    output_path,
    exist_ok=True
)

EVALUATION_GAMES = 1000
EVALUATION_SEED = 42

BAR_COLOR = '#4c78a8'
HIGHLIGHT_BAR_COLOR = '#f58518'


def adjust_y_axis(values):
    min_value = min(values)
    max_value = max(values)

    if min_value >= 0.0:
        plt.ylim(0.0, max_value * 1.2 if max_value > 0.0 else 1.0)
        return

    if max_value <= 0.0:
        plt.ylim(min_value * 1.2, 0.0)
        return

    value_range = max_value - min_value
    plt.ylim(
        min_value - 0.2 * value_range,
        max_value + 0.2 * value_range
    )


def add_bar_value_labels(bars):
    y_min, y_max = plt.ylim()
    y_offset = 0.01 * (y_max - y_min)

    for bar in bars:
        value = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2.0

        if value >= 0.0:
            y = value + y_offset
            vertical_alignment = 'bottom'
        else:
            y = value - y_offset
            vertical_alignment = 'top'

        plt.text(
            x,
            y,
            f'{value:.2f}',
            ha='center',
            va=vertical_alignment,
            fontsize=10.0
        )


def bar_colors(values, highlight_mode=None):
    if not highlight_mode:
        return [BAR_COLOR for _ in values]

    highlighted_value = max(values) if highlight_mode == 'max' else min(values)
    return [
        HIGHLIGHT_BAR_COLOR if value == highlighted_value else BAR_COLOR
        for value in values
    ]


def save_agent_metric_bar_plot(file_name, title, y_label, values, highlight_mode=None):
    figure = plt.figure(figsize=(12, 6))
    agent_names = [evaluator.model_name for evaluator in evaluators]

    bars = plt.bar(
        agent_names,
        values,
        color=bar_colors(values, highlight_mode)
    )
    adjust_y_axis(values)
    add_bar_value_labels(bars)

    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=15.0, ha='right')
    plt.tight_layout()

    figure.savefig(
        os.path.join(output_path, file_name)
    )
    plt.close(figure)


def save_lane_metric_bar_plot(file_name, title, y_label, values):
    figure = plt.figure(figsize=(16, 6))
    lane_indices = np.arange(len(LANE_NAMES))

    bars = plt.bar(
        lane_indices,
        values,
        color=bar_colors(values)
    )
    adjust_y_axis(values)
    add_bar_value_labels(bars)

    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(lane_indices, LANE_NAMES, rotation=35.0, ha='right')
    plt.tight_layout()

    figure.savefig(
        os.path.join(output_path, file_name)
    )
    plt.close(figure)


def save_evaluation_plots():
    save_agent_metric_bar_plot(
        'average_game_length.jpg',
        'Average Game Length',
        'Ticks',
        [evaluator.average_ticks for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'average_reward.jpg',
        'Average Reward',
        'Reward',
        [evaluator.average_reward for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'collision_percentage.jpg',
        'Collision Percentage',
        'Collisions (%)',
        [evaluator.collision_percentage * 100.0 for evaluator in evaluators],
        'min'
    )

    save_agent_metric_bar_plot(
        'total_vehicles_passed.jpg',
        'Total Vehicles Passed',
        'Vehicles',
        [evaluator.total_passed_vehicle_count for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'average_vehicles_passed.jpg',
        'Average Vehicles Passed',
        'Vehicles',
        [evaluator.average_passed_vehicle_count for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'max_vehicles_passed.jpg',
        'Max Vehicles Passed',
        'Vehicles',
        [evaluator.max_passed_vehicle_count for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'average_active_vehicles.jpg',
        'Average Active Vehicles',
        'Vehicles',
        [evaluator.average_active_vehicles for evaluator in evaluators]
    )

    save_agent_metric_bar_plot(
        'max_active_vehicles.jpg',
        'Max Active Vehicles',
        'Vehicles',
        [evaluator.max_active_vehicles for evaluator in evaluators]
    )

    save_agent_metric_bar_plot(
        'total_cars_passed.jpg',
        'Total Cars Passed',
        'Cars',
        [evaluator.total_passed_car_count for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'average_cars_passed.jpg',
        'Average Cars Passed',
        'Cars',
        [evaluator.average_passed_car_count for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'total_trains_passed.jpg',
        'Total Trains Passed',
        'Trains',
        [evaluator.total_passed_train_count for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'average_trains_passed.jpg',
        'Average Trains Passed',
        'Trains',
        [evaluator.average_passed_train_count for evaluator in evaluators],
        'max'
    )

    save_agent_metric_bar_plot(
        'average_train_waiting_time.jpg',
        'Average Waiting Time (Trains)',
        'Ticks',
        [evaluator.train_average_waiting_ticks for evaluator in evaluators],
        'min'
    )

    save_agent_metric_bar_plot(
        'average_car_waiting_time.jpg',
        'Average Waiting Time (Cars)',
        'Ticks',
        [evaluator.car_average_waiting_ticks for evaluator in evaluators],
        'min'
    )

    save_agent_metric_bar_plot(
        'average_phase_changes.jpg',
        'Average Phase Changes',
        'Phase Changes',
        [evaluator.average_phase_changes for evaluator in evaluators],
        'min'
    )

    save_lane_metric_bar_plot(
        'average_queue_length_random_agent.jpg',
        'Random Agent - Average Queue Length',
        'Average Queue Length',
        random_agent_evaluator.average_queue_length
    )

    save_lane_metric_bar_plot(
        'max_queue_length_random_agent.jpg',
        'Random Agent - Max Queue Length',
        'Max Queue Length',
        random_agent_evaluator.max_queue_length
    )

    save_lane_metric_bar_plot(
        'queue_active_percentage_random_agent.jpg',
        'Random Agent - Queue Active Percentage',
        'Queue Active (%)',
        [queue_active * 100.0 for queue_active in random_agent_evaluator.queue_active_percentage]
    )

    save_lane_metric_bar_plot(
        'green_light_percentage_random_agent.jpg',
        'Random Agent - Green Light Percentage',
        'Green Light (%)',
        [green_light * 100.0 for green_light in random_agent_evaluator.green_light_percentage]
    )

    save_lane_metric_bar_plot(
        'average_queue_length_basic_agent.jpg',
        'Basic Agent - Average Queue Length',
        'Average Queue Length',
        basic_agent_evaluator.average_queue_length
    )

    save_lane_metric_bar_plot(
        'max_queue_length_basic_agent.jpg',
        'Basic Agent - Max Queue Length',
        'Max Queue Length',
        basic_agent_evaluator.max_queue_length
    )

    save_lane_metric_bar_plot(
        'queue_active_percentage_basic_agent.jpg',
        'Basic Agent - Queue Active Percentage',
        'Queue Active (%)',
        [queue_active * 100.0 for queue_active in basic_agent_evaluator.queue_active_percentage]
    )

    save_lane_metric_bar_plot(
        'green_light_percentage_basic_agent.jpg',
        'Basic Agent - Green Light Percentage',
        'Green Light (%)',
        [green_light * 100.0 for green_light in basic_agent_evaluator.green_light_percentage]
    )

    save_lane_metric_bar_plot(
        'average_queue_length_dqn_agent.jpg',
        'DQN Agent - Average Queue Length',
        'Average Queue Length',
        dqn_agent_evaluator.average_queue_length
    )

    save_lane_metric_bar_plot(
        'max_queue_length_dqn_agent.jpg',
        'DQN Agent - Max Queue Length',
        'Max Queue Length',
        dqn_agent_evaluator.max_queue_length
    )

    save_lane_metric_bar_plot(
        'queue_active_percentage_dqn_agent.jpg',
        'DQN Agent - Queue Active Percentage',
        'Queue Active (%)',
        [queue_active * 100.0 for queue_active in dqn_agent_evaluator.queue_active_percentage]
    )

    save_lane_metric_bar_plot(
        'green_light_percentage_dqn_agent.jpg',
        'DQN Agent - Green Light Percentage',
        'Green Light (%)',
        [green_light * 100.0 for green_light in dqn_agent_evaluator.green_light_percentage]
    )

    save_lane_metric_bar_plot(
        'average_queue_length_ppo_agent.jpg',
        'PPO Agent - Average Queue Length',
        'Average Queue Length',
        ppo_agent_evaluator.average_queue_length
    )

    save_lane_metric_bar_plot(
        'max_queue_length_ppo_agent.jpg',
        'PPO Agent - Max Queue Length',
        'Max Queue Length',
        ppo_agent_evaluator.max_queue_length
    )

    save_lane_metric_bar_plot(
        'queue_active_percentage_ppo_agent.jpg',
        'PPO Agent - Queue Active Percentage',
        'Queue Active (%)',
        [queue_active * 100.0 for queue_active in ppo_agent_evaluator.queue_active_percentage]
    )

    save_lane_metric_bar_plot(
        'green_light_percentage_ppo_agent.jpg',
        'PPO Agent - Green Light Percentage',
        'Green Light (%)',
        [green_light * 100.0 for green_light in ppo_agent_evaluator.green_light_percentage]
    )


random_agent_evaluator = RandomAgentEvaluator(
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

basic_agent_evaluator = BasicAgentEvaluator(
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

dqn_agent_evaluator = DQNAgentEvaluator(
    model_path='training_output/dqn/1782752102/model.pt',
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

ppo_agent_evaluator = PPOAgentEvaluator(
    model_path='training_output/ppo/1782736882/model.pt',
    target_games=EVALUATION_GAMES,
    seed=EVALUATION_SEED
)

evaluators: list[AgentEvaluator] = [
    random_agent_evaluator,
    basic_agent_evaluator,
    dqn_agent_evaluator,
    ppo_agent_evaluator
]

for e in evaluators:
    e.run()
    print('\n\n\n')
    e.end()

save_evaluation_plots()
