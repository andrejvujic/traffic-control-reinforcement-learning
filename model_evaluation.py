from abc import ABC, abstractmethod
import pygame
from src.environment.traffic.vehicle_service import VehicleService
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, TRAFFIC_LIGHT_PHASES, TOTAL_LANES, LANE_NAMES
from src.game.map import Map
import random
import time


class ModelEvaluation:
    def __init__(
        self,
        model_name,
        target_games,
        seed
    ):
        random.seed(None)
        random.seed(seed)

        self.model_name = model_name
        self.target_games = target_games

        pygame.display.init()
        pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.traffic_light_service = TrafficLightService()
        self.vehicle_service = VehicleService(
            traffic_light_service=self.traffic_light_service
        )
        self.map = Map()

    @abstractmethod
    def select_action(self, state):
        pass

    def run(self):
        start_time = time.time()

        average_ticks, average_reward, collision_percentage, total_passed_vehicle_count, \
            average_passed_vehicle_count, max_passed_vehicle_count, total_passed_car_count, \
            average_passed_car_count, total_passed_train_count, average_passed_train_count, average_flow_rate, \
            max_flow_rate, average_phase_changes, average_queue_length, max_queue_length, queue_active_percentage, \
            train_average_waiting_ticks, car_average_waiting_ticks\
            = self.__run()

        evaluation_duration = time.time() - start_time
        print(f'\nEvaluation Took ->', end=' ')
        self.__print_formatted_duration(evaluation_duration)

        print('\n')

        print(f'Model -> {self.model_name}')

        print(f'Average Game Length -> {average_ticks} ticks')
        print(f'Average Reward -> {average_reward:.2f}')
        print(f'Collisions -> {collision_percentage * 100.0:.0f}%\n')
        print(f'Total Vehicles Passed -> {total_passed_vehicle_count}')
        print(f'Average Vehicles Passed -> {average_passed_vehicle_count}')
        print(f'Max Vehicles Passed -> {max_passed_vehicle_count}\n')
        print(f'\tTotal Cars Passed -> {total_passed_car_count}')
        print(f'\tAverage Cars Passed -> {average_passed_car_count}')
        print(f'\tTotal Trains Passed -> {total_passed_train_count}')
        print(f'\tAverage Trains Passed -> {average_passed_train_count}\n')
        print(f'Average Waiting Time (Trains) -> {train_average_waiting_ticks:.2f} ticks')
        print(f'Average Waiting Time (Cars) -> {car_average_waiting_ticks:.2f} ticks')
        print(f'Average Flow Rate -> {average_flow_rate:.2f}')
        print(f'Max Flow Rate -> {max_flow_rate:.2f}\n')
        print(f'Average Phase Changes -> {average_phase_changes:.1f}\n')

        print('Lane Statistics:')

        for index, (avearage_length, max_length, queue_active) in enumerate(
            zip(average_queue_length, max_queue_length, queue_active_percentage)
        ):
            print(f'\t{LANE_NAMES[index]}')
            print(f'\t\t- Average Queue Length -> {avearage_length:.2f}')
            print(f'\t\t- Max Queue Length -> {max_length}')
            print(f'\t\t- Queue Active -> {queue_active * 100.0:.0f}%')

    def end(self):
        pygame.quit()

    def __run(self):
        total_ticks = 0
        total_reward = 0.0
        total_collisions = 0
        passed_vehicle_count = []
        passed_car_count = []
        passed_train_count = []
        flow_rate = []
        queue_length = [[] for _ in range(TOTAL_LANES)]
        phase_changes = []
        train_waiting_ticks = []
        car_waiting_ticks = []

        for game_index in range(self.target_games):
            game_ticks = 0
            game_reward = 0.0
            game_phase_changes = 0

            terminated_flag, truncated_flag = False, False

            self.traffic_light_service.reset()
            self.vehicle_service.reset()
            current_state = self.vehicle_service.state()

            while not terminated_flag and not truncated_flag:
                action = self.select_action(current_state)

                if action < len(TRAFFIC_LIGHT_PHASES):
                    self.traffic_light_service.apply_phase(action)
                else:
                    self.traffic_light_service.turn_all_red()

                current_state, reward, terminated_flag, truncated_flag = self.vehicle_service.update()

                game_ticks = game_ticks + 1
                game_reward = game_reward + reward

                if not self.traffic_light_service.did_keep_phase():
                    game_phase_changes = game_phase_changes + 1

                if terminated_flag:
                    total_collisions = total_collisions + 1

            for index, queue_length_history in enumerate(self.vehicle_service.queue_length_history):
                queue_length[index].extend(queue_length_history)

            evaluation_progress = (game_index + 1) / self.target_games
            print(
                f"\rEvaluating {self.model_name}... Progress -> {evaluation_progress * 100.0:.0f}%",
                end='',
                flush=True
            )

            total_ticks = total_ticks + game_ticks
            total_reward = total_reward + game_reward

            total_vehicles_passed = self.vehicle_service.total_cars_passed + self.vehicle_service.total_trains_passed
            passed_vehicle_count.append(total_vehicles_passed)
            passed_car_count.append(self.vehicle_service.total_cars_passed)
            passed_train_count.append(self.vehicle_service.total_trains_passed)

            flow_rate.append(
                total_vehicles_passed / game_ticks if game_ticks > 0 else 0.0
            )

            phase_changes.append(game_phase_changes)
            train_waiting_ticks.append(
                self.vehicle_service.total_ticks_waiting_trains()
            )
            car_waiting_ticks.append(
                self.vehicle_service.total_ticks_waiting_cars()
            )

        max_queue_length = []
        average_queue_length = []

        for queue_length_history in queue_length:
            max_queue_length.append(
                max(queue_length_history)
            )

            average_queue_length.append(
                self.__calculate_average(queue_length_history)
            )

        queue_active_percentage = []
        for queue_length_history in queue_length:
            total_samples = len(queue_length_history)
            queue_active_percentage.append(
                len([waiting_vehicles for waiting_vehicles in queue_length_history if waiting_vehicles > 0])
                / total_samples if total_samples > 0 else 0.0
            )

        average_phase_changes = self.__calculate_average(phase_changes)

        average_flow_rate = self.__calculate_average(flow_rate)
        max_flow_rate = max(flow_rate)

        total_passed_vehicle_count = sum(passed_vehicle_count)
        average_passed_vehicle_count = self.__calculate_average(passed_vehicle_count)
        max_passed_vehicle_count = max(passed_vehicle_count)

        total_passed_car_count = sum(passed_car_count)
        average_passed_car_count = self.__calculate_average(passed_car_count)

        total_passed_train_count = sum(passed_train_count)
        average_passed_train_count = self.__calculate_average(passed_train_count)

        average_ticks = total_ticks / self.target_games

        train_average_waiting_ticks = sum(train_waiting_ticks) / total_passed_train_count if total_passed_train_count > 0 else 0.0
        car_average_waiting_ticks = sum(car_waiting_ticks) / total_passed_car_count if total_passed_car_count > 0 else 0.0

        average_reward = total_reward / self.target_games
        collision_percentage = total_collisions / self.target_games

        return average_ticks, average_reward, collision_percentage, total_passed_vehicle_count, \
            average_passed_vehicle_count, max_passed_vehicle_count, total_passed_car_count, average_passed_car_count, \
            total_passed_train_count, average_passed_train_count, average_flow_rate, max_flow_rate, average_phase_changes, \
            average_queue_length, max_queue_length, queue_active_percentage, train_average_waiting_ticks, car_average_waiting_ticks

    def __calculate_average(self, values):
        return sum(values) / len(values) if len(values) > 0 else 0.0

    def __print_formatted_duration(self, duration):
        duration_minutes = int(duration / 60)
        duration_seconds = int(duration - 60 * duration_minutes)
        if duration_minutes > 0:
            print(f'{duration_minutes} min', end=' ')
        print(f'{duration_seconds} seconds')
