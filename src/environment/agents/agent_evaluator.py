from abc import ABC, abstractmethod
import pygame
from src.environment.traffic.vehicles.vehicle_service import VehicleService
from src.environment.traffic.traffic_lights.traffic_light_service import TrafficLightService
from src.game.constants import CANVAS_WIDTH, CANVAS_HEIGHT, TRAFFIC_LIGHT_PHASES, TOTAL_LANE_COUNT, LANE_NAMES
from src.game.map import Map
import random
import time


class AgentEvaluator(ABC):
    def __init__(
        self,
        model_name,
        target_games,
        seed
    ):
        random.seed()
        random.seed(seed)

        self.model_name = model_name
        self.target_games = target_games

        pygame.display.init()
        pygame.display.set_mode(
            (CANVAS_WIDTH, CANVAS_HEIGHT)
        )

        self.traffic_light_service = TrafficLightService()
        self.vehicle_service = VehicleService(
            traffic_light_service=self.traffic_light_service
        )
        self.map = Map()

        self.average_ticks = 0.0
        self.average_reward = 0.0
        self.collision_percentage = 0.0

        self.total_passed_vehicle_count = 0
        self.average_passed_vehicle_count = 0.0
        self.max_passed_vehicle_count = 0

        self.average_active_vehicles = 0.0
        self.max_active_vehicles = 0

        self.total_passed_car_count = 0
        self.average_passed_car_count = 0.0
        self.total_passed_train_count = 0
        self.average_passed_train_count = 0.0

        self.train_average_waiting_ticks = 0.0
        self.car_average_waiting_ticks = 0.0

        self.average_phase_changes = 0.0

        self.average_queue_length = []
        self.max_queue_length = []
        self.queue_active_percentage = []
        self.green_light_percentage = []
        self.passed_vehicle_count_per_lane = []

    @abstractmethod
    def select_action(self, state):
        pass

    def run(self):
        start_time = time.time()

        self.__run()

        evaluation_duration = time.time() - start_time
        print(f'\nEvaluation Took ->', end=' ')
        self.__print_formatted_duration(evaluation_duration)

        print('\n')

        print(f'Model -> {self.model_name}')

        print(f'Average Game Length -> {self.average_ticks} ticks')
        print(f'Average Reward -> {self.average_reward:.2f}')
        print(f'Collisions -> {self.collision_percentage * 100.0:.0f}%\n')
        print(f'Total Vehicles Passed -> {self.total_passed_vehicle_count}')
        print(f'Average Vehicles Passed -> {self.average_passed_vehicle_count}')
        print(f'Max Vehicles Passed -> {self.max_passed_vehicle_count}')
        print(f'Average Active Vehicles -> {self.average_active_vehicles:.2f}')
        print(f'Max Active Vehicles -> {self.max_active_vehicles}\n')
        print(f'\tTotal Cars Passed -> {self.total_passed_car_count}')
        print(f'\tAverage Cars Passed -> {self.average_passed_car_count}')
        print(f'\tTotal Trains Passed -> {self.total_passed_train_count}')
        print(f'\tAverage Trains Passed -> {self.average_passed_train_count}\n')
        print(f'Average Waiting Time (Trains) -> {self.train_average_waiting_ticks:.2f} ticks')
        print(f'Average Waiting Time (Cars) -> {self.car_average_waiting_ticks:.2f} ticks')
        print(f'Average Phase Changes -> {self.average_phase_changes:.1f}\n')

        print('Lane Statistics:')

        for index, (average_length, max_length, queue_active, green_light, passed_count) in enumerate(
            zip(
                self.average_queue_length,
                self.max_queue_length,
                self.queue_active_percentage,
                self.green_light_percentage,
                self.passed_vehicle_count_per_lane
            )
        ):
            print(f'\t{LANE_NAMES[index]}')
            print(f'\t\t- Average Queue Length -> {average_length:.2f}')
            print(f'\t\t- Max Queue Length -> {max_length}')
            print(f'\t\t- Queue Active -> {queue_active * 100.0:.0f}%')
            print(f'\t\t- Green Light -> {green_light * 100.0:.0f}%')
            print(f'\t\t- Vehicles Passed -> {passed_count}')

    def end(self):
        pygame.display.quit()
        pygame.quit()

    def __run(self):
        total_ticks = 0
        total_reward = 0.0
        total_collisions = 0
        passed_vehicle_count = []
        passed_car_count = []
        passed_train_count = []
        queue_length = [[] for _ in range(TOTAL_LANE_COUNT)]
        phase_changes = []
        train_waiting_ticks = []
        car_waiting_ticks = []
        train_count = []
        car_count = []
        total_active_vehicles = 0
        max_active_vehicles = 0
        green_light_ticks = [0 for _ in range(TOTAL_LANE_COUNT)]
        passed_vehicle_count_per_lane = [0 for _ in range(TOTAL_LANE_COUNT)]

        total_duration = 0.0
        for game_index in range(self.target_games):

            game_ticks = 0
            game_reward = 0.0
            game_phase_changes = 0

            terminated_flag, truncated_flag = False, False

            self.traffic_light_service.reset()
            self.vehicle_service.reset()
            current_state = self.vehicle_service.state()

            game_start = time.time()
            while not terminated_flag and not truncated_flag:
                action = self.select_action(current_state)

                if action < len(TRAFFIC_LIGHT_PHASES):
                    self.traffic_light_service.apply_phase(action)
                else:
                    self.traffic_light_service.turn_all_red()

                for lane_index, is_passable in enumerate(self.traffic_light_service.state()):
                    if is_passable:
                        green_light_ticks[lane_index] = green_light_ticks[lane_index] + 1

                current_state, reward, terminated_flag, truncated_flag = self.vehicle_service.update()

                game_ticks = game_ticks + 1
                game_reward = game_reward + reward

                if not self.traffic_light_service.did_keep_phase():
                    game_phase_changes = game_phase_changes + 1

                if terminated_flag:
                    total_collisions = total_collisions + 1

                active_vehicles = self.vehicle_service.active_vehicles()
                total_active_vehicles = total_active_vehicles + active_vehicles
                if active_vehicles > max_active_vehicles:
                    max_active_vehicles = active_vehicles

            total_duration = total_duration + time.time() - game_start

            for index, queue_length_history in enumerate(self.vehicle_service.queue_length_history):
                queue_length[index].extend(queue_length_history)

            for index, passed_count in enumerate(self.vehicle_service.passed_vehicle_count_per_lane):
                passed_vehicle_count_per_lane[index] = passed_vehicle_count_per_lane[index] + passed_count

            games_done = game_index + 1
            games_left = self.target_games - games_done
            evaluation_progress = games_done / self.target_games

            average_duration = total_duration / games_done
            expected_duration = average_duration * games_left
            print(
                f"\rEvaluating {self.model_name}... Progress -> {evaluation_progress * 100.0:.0f}% | Approximated Time Left -> {int(expected_duration):5d} seconds",
                end='',
                flush=True
            )

            total_ticks = total_ticks + game_ticks
            total_reward = total_reward + game_reward

            total_vehicles_passed = self.vehicle_service.total_cars_passed + self.vehicle_service.total_trains_passed
            passed_vehicle_count.append(total_vehicles_passed)
            passed_car_count.append(self.vehicle_service.total_cars_passed)
            passed_train_count.append(self.vehicle_service.total_trains_passed)

            phase_changes.append(game_phase_changes)
            train_waiting_ticks.append(
                self.vehicle_service.total_ticks_waiting_trains()
            )
            car_waiting_ticks.append(
                self.vehicle_service.total_ticks_waiting_cars()
            )
            train_count.append(
                self.vehicle_service.total_trains_passed + len(self.vehicle_service.trains)
            )
            car_count.append(
                self.vehicle_service.total_cars_passed + len(self.vehicle_service.cars)
            )

        self.average_active_vehicles = total_active_vehicles / total_ticks if total_ticks > 0 else 0.0
        self.max_active_vehicles = max_active_vehicles
        self.green_light_percentage = [
            green_ticks / total_ticks if total_ticks > 0 else 0.0
            for green_ticks in green_light_ticks
        ]
        self.passed_vehicle_count_per_lane = passed_vehicle_count_per_lane

        self.max_queue_length = []
        self.average_queue_length = []

        for queue_length_history in queue_length:
            self.max_queue_length.append(
                max(queue_length_history)
            )

            self.average_queue_length.append(
                self.__calculate_average(queue_length_history)
            )

        self.queue_active_percentage = []
        for queue_length_history in queue_length:
            total_samples = len(queue_length_history)
            self.queue_active_percentage.append(
                len([waiting_vehicles for waiting_vehicles in queue_length_history if waiting_vehicles > 0])
                / total_samples if total_samples > 0 else 0.0
            )

        self.average_phase_changes = self.__calculate_average(phase_changes)

        self.total_passed_vehicle_count = sum(passed_vehicle_count)
        self.average_passed_vehicle_count = self.__calculate_average(passed_vehicle_count)
        self.max_passed_vehicle_count = max(passed_vehicle_count)

        self.total_passed_car_count = sum(passed_car_count)
        self.average_passed_car_count = self.__calculate_average(passed_car_count)

        self.total_passed_train_count = sum(passed_train_count)
        self.average_passed_train_count = self.__calculate_average(passed_train_count)

        self.average_ticks = total_ticks / self.target_games

        total_train_count = sum(train_count)
        total_car_count = sum(car_count)

        self.train_average_waiting_ticks = sum(train_waiting_ticks) / total_train_count if total_train_count > 0 else 0.0
        self.car_average_waiting_ticks = sum(car_waiting_ticks) / total_car_count if total_car_count > 0 else 0.0

        self.average_reward = total_reward / self.target_games
        self.collision_percentage = total_collisions / self.target_games

    def __calculate_average(self, values):
        return sum(values) / len(values) if len(values) > 0 else 0.0

    def __print_formatted_duration(self, duration):
        duration_minutes = int(duration / 60)
        duration_seconds = int(duration - 60 * duration_minutes)
        if duration_minutes > 0:
            print(f'{duration_minutes} min', end=' ')
        print(f'{duration_seconds} seconds')
