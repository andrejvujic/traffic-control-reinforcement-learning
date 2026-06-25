from src.environment.traffic.car import Car
from src.environment.traffic.car_spawner import CarSpawner
from src.environment.traffic.train import Train
from src.environment.traffic.train_spawner import TrainSpawner
from src.environment.traffic.traffic_light_service import TrafficLightService

from src.game.constants import TRAFFIC_LIGHT_PHASES, MAX_TICKS_PER_EPISODE
from src.game.constants import VEHICLE_STOP_POSITIONS, CAR_LANES, TRAIN_LANES, TOTAL_LANES
from src.game.constants import CAR_MOVEMENT_INTERVAL, CAR_SPAWN_INTERVAL, CAR_SPAWN_PROBABILITY
from src.game.constants import TRAIN_MOVEMENT_INTERVAL, TRAIN_SPAWN_INTERVAL, TRAIN_SPAWN_PROBABILITY
from src.game.utilities import random_bool

import pygame


class VehicleService:
    VEHICLE_COUNT_NORMALIZER = 15

    def __init__(self, traffic_light_service: TrafficLightService):
        self.traffic_light_service = traffic_light_service
        self.ticks = 0

        self.total_cars_passed = 0
        self.cars_passed_this_tick = 0
        self.total_ticks_waiting_cars = 0

        self.cars: list[Car] = []
        self.car_spawner = CarSpawner(assets_directory='assets/cars')

        self.total_trains_passed = 0
        self.trains_passed_this_tick = 0
        self.total_ticks_waiting_trains = 0

        self.trains: list[Train] = []
        self.train_spawner = TrainSpawner(assets_directory='assets/trains')

        self.queue_length = [0 for _ in range(TOTAL_LANES)]

    def reset(self):
        self.cars = []
        self.trains = []
        self.ticks = 0

        self.total_cars_passed = 0
        self.cars_passed_this_tick = 0
        self.total_ticks_waiting_cars = 0

        self.total_trains_passed = 0
        self.trains_passed_this_tick = 0
        self.total_ticks_waiting_trains = 0

        self.queue_length = [0 for _ in range(TOTAL_LANES)]

    def update(self):
        self.ticks = self.ticks + 1
        self.cars_passed_this_tick = 0
        self.trains_passed_this_tick = 0
        self.queue_length = [0 for _ in range(TOTAL_LANES)]

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

        for car in self.cars:
            if self.ticks % CAR_MOVEMENT_INTERVAL == 0 and self.can_car_move(car):
                car.move()
                continue

            car.mark_waiting()
            self.queue_length[car.lane_index] += 1

        for train in self.trains:
            if self.ticks % TRAIN_MOVEMENT_INTERVAL == 0 and self.can_train_move(train):
                train.move()
                continue

            train.mark_waiting()
            self.queue_length[train.lane_index] += 1

        self.__remove_dead_vehicles()

        return self.state(), self.evaluate_state(), self.has_collision(), self.ticks >= MAX_TICKS_PER_EPISODE

    def can_car_move(self, car: Car):
        stop_position = VEHICLE_STOP_POSITIONS[car.lane_index]
        sibling_cars = self.__vehicles_in_lane(car.lane_index)

        for sibling_car in sibling_cars:
            if car.next_position() == sibling_car.position():
                return False

        must_stop = car.position() == stop_position and not self.traffic_light_service.is_passable(car.lane_index)
        return not must_stop

    def can_train_move(self, train: Train):
        stop_position = VEHICLE_STOP_POSITIONS[train.lane_index]
        must_stop = train.position() == stop_position and not self.traffic_light_service.is_passable(train.lane_index)
        return not must_stop

    def has_collision(self):
        vehicles = [*self.cars, *self.trains]
        for vehicle in vehicles:
            if vehicle.collides_with_any(vehicles):
                return True

        return False

    def __remove_dead_vehicles(self):
        for vehicle in [*self.cars, *self.trains]:
            if not vehicle.on_screen():
                if isinstance(vehicle, Car):
                    self.total_cars_passed = self.total_cars_passed + 1
                    self.cars_passed_this_tick = self.cars_passed_this_tick + 1
                    self.total_ticks_waiting_cars = self.total_ticks_waiting_cars + vehicle.ticks_waiting

                if isinstance(vehicle, Train):
                    self.total_trains_passed = self.total_trains_passed + 1
                    self.trains_passed_this_tick = self.trains_passed_this_tick + 1
                    self.total_ticks_waiting_trains = self.total_ticks_waiting_trains + vehicle.ticks_waiting

        self.cars = [car for car in self.cars if car.on_screen()]
        self.trains = [train for train in self.trains if train.on_screen()]

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
                for lane_index in range(TOTAL_LANES)
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

    def __passing_cars(self):
        return sum([self.__count_passing(lane_index) for lane_index in CAR_LANES])

    def __passing_trains(self):
        return sum([self.__count_passing(lane_index) for lane_index in TRAIN_LANES])

    def __approaching_trains(self):
        return sum([self.__count_approaching(lane_index) for lane_index in TRAIN_LANES])

    def evaluate_state(self):
        reward = 0.0

        # Small positive reward for avoiding surviving.
        reward = reward + 0.05

        if self.has_collision():
            reward = reward - 600.0

        appoarching_vehicles_per_lane = [
            self.__count_approaching(lane_index) for lane_index in range(TOTAL_LANES)
        ]
        passing_vehicles_per_lane = [
            self.__count_passing(lane_index) for lane_index in range(TOTAL_LANES)
        ]

        total_passing_cars = self.__passing_cars()
        total_passing_trains = self.__passing_trains()

        old_traffic_light_state = self.traffic_light_service.previous_state
        new_traffic_light_state = self.traffic_light_service.state()

        for lane_index, (previously_passable, now_passable) in enumerate(
            zip(old_traffic_light_state, new_traffic_light_state)
        ):
            # If the agent changes the state of the traffic lights
            # and there are still vehicles of the opposite type
            # passing through the intersection it gets a negative reward.
            if not previously_passable and now_passable:
                if lane_index in CAR_LANES and total_passing_trains > 0:
                    reward = reward - 6.0

                if lane_index in TRAIN_LANES and total_passing_cars > 0:
                    reward = reward - 3.0

                # The agent gets a negative reward for enabling
                # traffic on incompatible car lanes.
                for other_lane_index in CAR_LANES:
                    is_compatible = {other_lane_index, lane_index} in TRAFFIC_LIGHT_PHASES
                    if passing_vehicles_per_lane[other_lane_index] > 0 and not is_compatible:
                        reward = reward - 0.25

        # The agents a small positive reward for keeping the same
        # traffic light phase between ticks.
        if self.traffic_light_service.previous_action_repeated():
            reward = reward + 0.1

        # Small negative reward for changing traffic light phase
        # while vehicles are still passing.
        if not self.traffic_light_service.previous_action_repeated():
            total_passing_vehicles = total_passing_cars + total_passing_trains
            total_passing_vehicles = total_passing_vehicles - 1
            if total_passing_vehicles > 0:
                reward = reward - 1.0

        for lane_index, is_passable in enumerate(new_traffic_light_state):
            if lane_index in TRAIN_LANES:
                continue

            # Negative reward for having traffic for cars
            # enabled when a train is approaching or waiting.
            if is_passable and self.__approaching_trains() > 0:
                reward = reward - 0.2

        for lane_index, (waiting_vehicles, passable) in enumerate(
            zip(
                appoarching_vehicles_per_lane,
                new_traffic_light_state
            )
        ):
            # Positive reward for having traffic enabled
            # on lanes where vehicles are waiting.
            if passable and waiting_vehicles > 0:
                reward = reward + 0.05 * waiting_vehicles
                continue

            # Negative reward for having traffic disabled
            # on lanes where vehicles are waiting.
            if not passable and waiting_vehicles > 0:
                if lane_index in CAR_LANES:
                    reward = reward - 0.25 * waiting_vehicles
                    continue

                # The reward is more negative for trains
                # to encourge the agent to give them priority.
                reward = reward - 2.0 * waiting_vehicles

        for lane_index, (passing_vehicles, passable) in enumerate(
            zip(
                passing_vehicles_per_lane,
                new_traffic_light_state
            )
        ):
            if not passing_vehicles:
                continue

            # Small positive reward for having vehicles
            # inside of the intersection.
            reward = reward + 0.025 * passing_vehicles

            # Negative reward for disabling traffic on a lane
            # where vehicles are still inside the intersection.
            if not passable:
                reward = reward - 1.0 * passing_vehicles
                continue

            # Positive reward for keeping traffic enabled
            # while vehiclesa are the intersection.
            if lane_index in CAR_LANES:
                reward = reward + 0.5 * passing_vehicles
            else:
                # Bigger reward for trains because they make collisions
                # more likely while in the intersection because
                # of their size.
                reward = reward + 3.5 * passing_vehicles

        # Positive reward for vehicles which successfully
        # passed the intersection.
        reward = reward + 0.5 * self.cars_passed_this_tick
        reward = reward + 5.0 * self.trains_passed_this_tick

        return reward

    def average_ticks_waiting_cars(self):
        total_ticks_waiting = self.total_ticks_waiting_cars + sum(
            car.ticks_waiting for car in self.cars
        )

        total_cars = self.total_cars_passed + len(self.cars)

        return total_ticks_waiting / total_cars if total_cars > 0 else 0.0

    def average_ticks_waiting_trains(self):
        total_ticks_waiting = self.total_ticks_waiting_trains + sum(
            train.ticks_waiting for train in self.trains
        )

        total_trains = self.total_trains_passed + len(self.trains)

        return total_ticks_waiting / total_trains if total_trains > 0 else 0.0

    def active_vehicles(self):
        return len([*self.cars, *self.trains])

    def draw(self, surface: pygame.Surface):
        for car in self.cars:
            car.draw(surface)

        for train in self.trains:
            train.draw(surface)
