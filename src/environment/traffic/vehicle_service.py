from src.environment.traffic.car import Car
from src.environment.traffic.car_spawner import CarSpawner
from src.environment.traffic.train import Train
from src.environment.traffic.train_spawner import TrainSpawner
from src.environment.traffic.traffic_light_service import TrafficLightService

from src.game.constants import MAX_TICKS_PER_EPISODE
from src.game.constants import VEHICLE_STOP_POSITIONS, CAR_LANES, TOTAL_LANES
from src.game.constants import CAR_MOVEMENT_INTERVAL, CAR_SPAWN_INTERVAL, CAR_SPAWN_PROBABILITY
from src.game.constants import TRAIN_MOVEMENT_INTERVAL, TRAIN_SPAWN_INTERVAL, TRAIN_SPAWN_PROBABILITY
from src.game.utilities import random_bool

import pygame


class VehicleService:
    def __init__(self, traffic_light_service: TrafficLightService):
        self.traffic_light_service = traffic_light_service
        self.ticks = 0

        self.cars: list[Car] = []
        self.car_spawner = CarSpawner(assets_directory='assets/cars')

        self.trains: list[Train] = []
        self.train_spawner = TrainSpawner(assets_directory='assets/trains')

    def reset(self):
        self.cars = []
        self.trains = []
        self.ticks = 0

    def update(self):
        self.ticks = self.ticks + 1

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

        for train in self.trains:
            if self.ticks % TRAIN_MOVEMENT_INTERVAL == 0 and self.can_train_move(train):
                train.move()

        self.__remove_dead_cars()
        self.__remove_dead_trains()

        return self.state(), self.evaluate_state(), self.has_collision(), self.ticks > MAX_TICKS_PER_EPISODE

    def can_car_move(self, car: Car):
        stop_position = VEHICLE_STOP_POSITIONS[car.lane_index]
        sibling_cars = self.__get_lane_vehicles(car.lane_index)

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

    def __remove_dead_cars(self):
        # TODO: This removes cars which had to be spawned outside of the screen...
        self.cars = [car for car in self.cars if car.is_alive()]

    def __remove_dead_trains(self):
        self.trains = [train for train in self.trains if train.is_alive()]

    def __get_lane_vehicles(self, lane_index):
        vehicles = [*self.cars, *self.trains]
        return [vehicle for vehicle in vehicles if vehicle.lane_index == lane_index]

    def __count_vehicles_in_intersection(self, lane_index):
        lane_vehicles = self.__get_lane_vehicles(lane_index)
        return len([vehicle for vehicle in lane_vehicles if vehicle.is_in_intersection()])

    def __count_vehicles_approaching(self, lane_index):
        lane_vehicles = self.__get_lane_vehicles(lane_index)
        return len([vehicle for vehicle in lane_vehicles if vehicle.is_approaching_intersection()])

    def __count_vehicles_passed(self, lane_index):
        lane_vehicles = self.__get_lane_vehicles(lane_index)
        return len([vehicle for vehicle in lane_vehicles if vehicle.has_passed_intersection()])

    def state(self):
        return [
            *[self.__count_vehicles_approaching(lane_index) for lane_index in range(TOTAL_LANES)],
            *[self.__count_vehicles_in_intersection(lane_index) for lane_index in range(TOTAL_LANES)],
        ]

    def evaluate_state(self):
        reward = 0.0

        if self.has_collision():
            reward = reward - 200.0

        traffic_light_states = self.traffic_light_service.state()
        appoarching_vehicles_per_lane = [self.__count_vehicles_approaching(lane_index) for lane_index in range(TOTAL_LANES)]

        for lane_index, (approaching_vehicle_count, is_traffic_light_passable) in enumerate(
            zip(
                appoarching_vehicles_per_lane,
                traffic_light_states
            )
        ):
            if not is_traffic_light_passable:
                if lane_index in CAR_LANES:
                    reward = reward - 0.5 * approaching_vehicle_count
                    continue

                reward = reward - 20.0

        passed_vehicles_per_lane = [self.__count_vehicles_passed(lane_index) for lane_index in range(TOTAL_LANES)]
        passed_vehicles_total = sum(passed_vehicles_per_lane)

        reward = reward + passed_vehicles_total * 10.0

        return reward

    def draw(self, surface: pygame.Surface):
        for car in self.cars:
            car.draw(surface)

        for train in self.trains:
            train.draw(surface)
