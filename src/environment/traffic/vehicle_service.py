from src.environment.traffic.car import Car
from src.environment.traffic.car_spawner import CarSpawner
from src.environment.traffic.train import Train
from src.environment.traffic.train_spawner import TrainSpawner
from src.environment.traffic.traffic_light_service import TrafficLightService

from src.game.constants import TRAFFIC_LIGHT_PHASES, MAX_TICKS_PER_EPISODE
from src.game.constants import VEHICLE_STOP_POSITIONS, CAR_LANES, TRAIN_LANES, ALL_LANES
from src.game.constants import CAR_MOVEMENT_INTERVAL, CAR_SPAWN_INTERVAL, CAR_SPAWN_PROBABILITY
from src.game.constants import TRAIN_MOVEMENT_INTERVAL, TRAIN_SPAWN_INTERVAL, TRAIN_SPAWN_PROBABILITY
from src.game.utilities import random_bool

import pygame


class VehicleService:
    VEHICLE_COUNT_NORMALIZER = 30
    REWARD_VEHICLE_COUNT_LIMIT = 20

    def __init__(self, traffic_light_service: TrafficLightService):
        self.traffic_light_service = traffic_light_service
        self.ticks = 0

        self.total_cars_passed = 0
        self.cars_passed_this_tick = 0
        self.passed_cars_ticks_waiting = 0

        self.cars: list[Car] = []
        self.car_spawner = CarSpawner(assets_directory='assets/cars')

        self.total_trains_passed = 0
        self.trains_passed_this_tick = 0
        self.passed_trains_ticks_waiting = 0

        self.trains: list[Train] = []
        self.train_spawner = TrainSpawner(assets_directory='assets/trains')

        self.queue_length_history = [[] for _ in ALL_LANES]

    def reset(self):
        self.cars = []
        self.trains = []
        self.ticks = 0

        self.total_cars_passed = 0
        self.cars_passed_this_tick = 0
        self.passed_cars_ticks_waiting = 0

        self.total_trains_passed = 0
        self.trains_passed_this_tick = 0
        self.passed_trains_ticks_waiting = 0

        self.queue_length_history = [[] for _ in ALL_LANES]

    def update(self):
        self.ticks = self.ticks + 1
        self.cars_passed_this_tick = 0
        self.trains_passed_this_tick = 0

        self.__spawn_new_vehicles()

        current_queue_length = [0 for _ in ALL_LANES]

        cars_in_lane = [
            self.__vehicles_in_lane(lane_index)
            for lane_index in CAR_LANES
        ]
        for car in self.cars:
            if self.ticks % CAR_MOVEMENT_INTERVAL == 0 and self.__can_car_move(car, cars_in_lane[car.lane_index]):
                car.move()
                continue

            car.mark_waiting()
            current_queue_length[car.lane_index] += 1

        for train in self.trains:
            if self.ticks % TRAIN_MOVEMENT_INTERVAL == 0 and self.__can_train_move(train):
                train.move()
                continue

            train.mark_waiting()
            current_queue_length[train.lane_index] += 1

        for index in ALL_LANES:
            self.queue_length_history[index].append(
                current_queue_length[index]
            )

        self.__remove_passed_vehicles()
        has_collision = self.has_collision()

        return self.state(), self.evaluate_state(has_collision), has_collision, self.ticks >= MAX_TICKS_PER_EPISODE

    def __can_car_move(self, car: Car, sibling_cars):
        stop_position = VEHICLE_STOP_POSITIONS[car.lane_index]

        for sibling_car in sibling_cars:
            if car.next_position() == sibling_car.position():
                return False

        must_stop = car.position() == stop_position and not self.traffic_light_service.is_passable(car.lane_index)
        return not must_stop

    def __can_train_move(self, train: Train):
        stop_position = VEHICLE_STOP_POSITIONS[train.lane_index]
        must_stop = train.position() == stop_position and not self.traffic_light_service.is_passable(train.lane_index)
        return not must_stop

    def has_collision(self):
        vehicles = [*self.cars, *self.trains]
        for vehicle in vehicles:
            if vehicle.collides_with_any(vehicles):
                return True

        return False

    def __spawn_new_vehicles(self):
        if self.ticks % CAR_SPAWN_INTERVAL == 0 and random_bool(CAR_SPAWN_PROBABILITY):
            self.cars.append(
                self.car_spawner.spawn(
                    existing_cars=self.cars
                )
            )

        if self.ticks % TRAIN_SPAWN_INTERVAL == 0 and random_bool(TRAIN_SPAWN_PROBABILITY):
            new_train = self.train_spawner.spawn(
                existing_trains=self.trains
            )

            if new_train:
                self.trains.append(new_train)

    def __remove_passed_vehicles(self):
        self.__update_statistics()

        self.cars = [car for car in self.cars if car.on_screen()]
        self.trains = [train for train in self.trains if train.on_screen()]

    def __update_statistics(self):
        for vehicle in [*self.cars, *self.trains]:
            if not vehicle.on_screen():
                if isinstance(vehicle, Car):
                    self.__mark_car_passed(vehicle)

                if isinstance(vehicle, Train):
                    self.__mark_train_passed(vehicle)

    def __mark_car_passed(self, car: Car):
        self.total_cars_passed = self.total_cars_passed + 1
        self.cars_passed_this_tick = self.cars_passed_this_tick + 1
        self.passed_cars_ticks_waiting = self.passed_cars_ticks_waiting + car.ticks_waiting

    def __mark_train_passed(self, train: Train):
        self.total_trains_passed = self.total_trains_passed + 1
        self.trains_passed_this_tick = self.trains_passed_this_tick + 1
        self.passed_trains_ticks_waiting = self.passed_trains_ticks_waiting + train.ticks_waiting

    def state(self):
        return [
            *self.traffic_light_service.state(),
            *[
                self.__normalize_vehicle_count(self.__count_approaching(lane_index))
                for lane_index in CAR_LANES
            ],
            *[
                self.__count_approaching(lane_index)
                for lane_index in TRAIN_LANES
            ],
            *[
                int(self.__count_passing(lane_index) > 0)
                for lane_index in ALL_LANES
            ],
            *[
                self.__normalize_vehicle_count(self.__count_passed(lane_index))
                for lane_index in CAR_LANES
            ],
            *[
                self.__count_passed(lane_index)
                for lane_index in TRAIN_LANES
            ],
        ]

    def __normalize_vehicle_count(self, vehicle_count):
        return min(
            vehicle_count,
            self.VEHICLE_COUNT_NORMALIZER
        ) / self.VEHICLE_COUNT_NORMALIZER

    def evaluate_state(self, has_collision):
        reward = 0.0

        if has_collision:
            reward = reward - 10000.0

        reward = reward + 0.01

        appoarching_vehicles_per_lane = [
            self.__count_approaching(lane_index) for lane_index in ALL_LANES
        ]
        passing_vehicles_per_lane = [
            self.__count_passing(lane_index) for lane_index in ALL_LANES
        ]

        total_passing_cars = self.__passing_cars()
        total_passing_trains = self.__passing_trains()

        old_traffic_light_state = self.traffic_light_service.previous_state
        new_traffic_light_state = self.traffic_light_service.state()

        for lane_index, (previously_passable, now_passable) in enumerate(
            zip(old_traffic_light_state, new_traffic_light_state)
        ):
            if not previously_passable and now_passable:
                if self.__did_make_safe_move(lane_index, total_passing_cars, total_passing_trains) and appoarching_vehicles_per_lane[lane_index] > 0:
                    if lane_index in CAR_LANES:
                        reward = reward + 7.0 + min(
                            appoarching_vehicles_per_lane[lane_index],
                            self.REWARD_VEHICLE_COUNT_LIMIT
                        )
                        continue

                    reward = reward + 120.0

        for lane_index, is_passable in enumerate(new_traffic_light_state):
            if not is_passable:
                continue

            has_approaching_vehicles = appoarching_vehicles_per_lane[lane_index] > 0
            has_passing_vehicles = passing_vehicles_per_lane[lane_index] > 0
            if has_approaching_vehicles or has_passing_vehicles:
                continue

            if lane_index in CAR_LANES:
                reward = reward - 3.5
                continue

            reward = reward - 40.0

        for lane_index, (waiting_vehicles, passable) in enumerate(
            zip(
                appoarching_vehicles_per_lane,
                new_traffic_light_state
            )
        ):
            if passable and waiting_vehicles > 0:
                if lane_index in CAR_LANES:
                    reward = reward + 2.5 * min(waiting_vehicles, self.REWARD_VEHICLE_COUNT_LIMIT)
                    continue

                reward = reward + 120.0 * waiting_vehicles

            if not passable and waiting_vehicles > 0:
                if lane_index in CAR_LANES:
                    reward = reward - 1.75 * min(waiting_vehicles, self.REWARD_VEHICLE_COUNT_LIMIT)
                    continue

                reward = reward - 100.0 * waiting_vehicles

        for lane_index, (passing_vehicles, passable) in enumerate(
            zip(
                passing_vehicles_per_lane,
                new_traffic_light_state
            )
        ):
            if not passing_vehicles:
                continue

            reward = reward + 1.0

            if not passable:
                reward = reward - 30.0
                continue

            if lane_index in CAR_LANES:
                reward = reward + 10.0
            else:
                reward = reward + 60.0

        reward = reward + 40.0 * self.cars_passed_this_tick
        reward = reward + 300.0 * self.trains_passed_this_tick

        return reward

    def average_ticks_waiting_cars(self):
        total_cars = self.total_cars_passed + len(self.cars)

        return self.total_ticks_waiting_cars() / total_cars if total_cars > 0 else 0.0

    def average_ticks_waiting_trains(self):
        total_trains = self.total_trains_passed + len(self.trains)
        return self.total_ticks_waiting_trains() / total_trains if total_trains > 0 else 0.0

    def total_ticks_waiting_cars(self):
        return self.passed_cars_ticks_waiting + sum(
            car.ticks_waiting for car in self.cars
        )

    def total_ticks_waiting_trains(self):
        return self.passed_trains_ticks_waiting + sum(
            train.ticks_waiting for train in self.trains
        )

    def active_vehicles(self):
        return len([*self.cars, *self.trains])

    def draw(self, surface: pygame.Surface):
        for car in self.cars:
            car.draw(surface)

        for train in self.trains:
            train.draw(surface)

    def __vehicles_in_lane(self, lane_index):
        vehicles = [*self.cars, *self.trains]
        return [vehicle for vehicle in vehicles if vehicle.lane_index == lane_index]

    def __count_passing(self, lane_index):
        lane_vehicles = self.__vehicles_in_lane(lane_index)
        return len([vehicle for vehicle in lane_vehicles if vehicle.is_in_intersection()])

    def __count_approaching(self, lane_index):
        lane_vehicles = self.__vehicles_in_lane(lane_index)
        return len([vehicle for vehicle in lane_vehicles if vehicle.is_approaching_intersection()])

    def __count_passed(self, lane_index):
        lane_vehicles = self.__vehicles_in_lane(lane_index)
        return len([vehicle for vehicle in lane_vehicles if vehicle.has_passed_intersection()])

    def __passing_cars(self):
        return sum([self.__count_passing(lane_index) for lane_index in CAR_LANES])

    def __passing_trains(self):
        return sum([self.__count_passing(lane_index) for lane_index in TRAIN_LANES])

    def __did_make_safe_move(self, lane_index, total_passing_cars, total_passing_trains):
        passing_vehicles_per_lane = [self.__count_passing(index) for index in ALL_LANES]
        for other_lane_index in CAR_LANES:
            is_compatible = {other_lane_index, lane_index} in TRAFFIC_LIGHT_PHASES
            if passing_vehicles_per_lane[other_lane_index] > 0 and not is_compatible:
                return False

        unsafe_for_cars = lane_index in CAR_LANES and total_passing_trains > 0
        unsafe_for_trains = lane_index in TRAIN_LANES and total_passing_cars > 0

        return not unsafe_for_cars and not unsafe_for_trains
