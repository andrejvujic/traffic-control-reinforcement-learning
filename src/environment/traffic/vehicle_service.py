from src.environment.traffic.car import Car
from src.environment.traffic.car_spawner import CarSpawner
from src.environment.traffic.train import Train
from src.environment.traffic.train_spawner import TrainSpawner
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.constants import VEHICLE_STOP_POSITIONS
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

    def can_car_move(self, car: Car):
        stop_position = VEHICLE_STOP_POSITIONS[car.lane_index]
        sibling_cars = self.__get_lane_cars(car.lane_index)

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

    def __get_lane_cars(self, lane_index):
        return [car for car in self.cars if car.lane_index == lane_index]

    def draw(self, surface: pygame.Surface):
        for car in self.cars:
            car.draw(surface)

        for train in self.trains:
            train.draw(surface)
