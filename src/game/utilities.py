import random
import os


def random_bool(true_probability=0.5):
    return random.choices(
        [True, False],
        weights=[true_probability, 1.0 - true_probability]
    )[0]


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


def save_history_plot(output_path, file_name, value_history, value_label, plot_title, window_size=None):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 6))

    if window_size:
        values = calculate_partial_moving_averages(
            value_history,
            window_size
        )
    else:
        values = value_history

    games = list(
        range(1, len(value_history) + 1)
    )

    plt.plot(
        games,
        values
    )

    plt.xlabel('Game')
    plt.ylabel(value_label)
    plt.title(plot_title)

    figure.savefig(
        os.path.join(output_path, file_name),
    )
    plt.close(figure)


def save_training_history_plots(
    output_path,
    moving_average_window_size,
    total_reward_history,
    tick_count_history,
    cars_waiting_ticks_history,
    trains_waiting_ticks_history,
    total_cars_passed_history,
    total_trains_passed_history,
    raw_output_path=None
):
    plot_configurations = [
        (
            'total_reward_history.jpg',
            total_reward_history,
            'Total Reward',
            'Total Reward History'
        ),
        (
            'game_length_history.jpg',
            tick_count_history,
            'Game Length (Ticks)',
            'Game Length History'
        ),
        (
            'car_ticks_waiting_history.jpg',
            cars_waiting_ticks_history,
            'Average Ticks Waiting',
            'Average Ticks Waiting History (Cars)'
        ),
        (
            'train_ticks_waiting_history.jpg',
            trains_waiting_ticks_history,
            'Average Ticks Waiting',
            'Average Ticks Waiting History (Trains)'
        ),
        (
            'total_cars_passed_history.jpg',
            total_cars_passed_history,
            'Cars Passed',
            'Cars Passed History'
        ),
        (
            'total_trains_passed_history.jpg',
            total_trains_passed_history,
            'Trains Passed',
            'Trains Passed History'
        ),
    ]

    if raw_output_path:
        for file_name, value_history, value_label, plot_title in plot_configurations:
            save_history_plot(
                raw_output_path,
                file_name,
                value_history,
                value_label,
                plot_title
            )

    for file_name, value_history, value_label, plot_title in plot_configurations:
        save_history_plot(
            output_path,
            file_name,
            value_history,
            value_label,
            f'{plot_title} ({moving_average_window_size}-Game PMA)',
            moving_average_window_size
        )


def save_agent_training_checkpoint(
    agent,
    model_path,
    output_path,
    moving_average_window_size,
    total_reward_history,
    tick_count_history,
    cars_waiting_ticks_history,
    trains_waiting_ticks_history,
    total_cars_passed_history,
    total_trains_passed_history,
    raw_output_path=None
):
    os.makedirs(
        output_path,
        exist_ok=True
    )

    agent.save(model_path)

    save_training_history_plots(
        output_path,
        moving_average_window_size,
        total_reward_history,
        tick_count_history,
        cars_waiting_ticks_history,
        trains_waiting_ticks_history,
        total_cars_passed_history,
        total_trains_passed_history,
        raw_output_path
    )


def log_training_message(output_path, message):
    print(message)
    with open(os.path.join(output_path, 'training_log.txt'), 'a') as file:
        file.write(f'{message}\n')


def append_training_history(
    vehicle_service,
    game_reward,
    game_ticks,
    cars_waiting_ticks_history,
    trains_waiting_ticks_history,
    total_cars_passed_history,
    total_trains_passed_history,
    total_reward_history,
    tick_count_history
):
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


def render_debug_frame(surface, map_, traffic_light_service, vehicle_service, clock):
    import pygame

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit

    surface.fill('black')

    map_.draw(surface)
    traffic_light_service.draw(surface)
    vehicle_service.draw(surface)

    pygame.display.flip()
    clock.tick()
