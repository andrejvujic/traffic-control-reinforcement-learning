from src.environment.traffic.car import Car
from src.environment.traffic.car_spawner import CarSpawner
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.constants import CAR_MOVEMENT_INTERVAL, CAR_SPAWN_INTERVAL, CAR_SPAWN_PROBABILITY, CAR_STOP_POSITIONS
from src.game.utilities import random_bool

import pygame


class VehicleService:
    def __init__(self, traffic_light_service: TrafficLightService):
        self.traffic_light_service = traffic_light_service
        self.ticks = 0
        self.cars: list[Car] = []

        self.car_spawner = CarSpawner(assets_directory='assets/cars')

    def reset(self):
        self.cars = []
        self.ticks = 0

    def update(self):
        self.ticks = self.ticks + 1

        if self.ticks % CAR_SPAWN_INTERVAL == 0 and random_bool(CAR_SPAWN_PROBABILITY):
            self.cars.append(
                self.car_spawner.spawn(
                    existing_cars=self.cars
                )
            )

        for car in self.cars:
            if self.ticks % CAR_MOVEMENT_INTERVAL == 0 and self.can_move(car):
                car.move()

        self.__remove_dead_cars()

    def can_move(self, car: Car):
        stop_position = CAR_STOP_POSITIONS[car.lane_index]
        sibling_cars = self.__get_lane_cars(car.lane_index)

        for sibling_car in sibling_cars:
            if car.next_position() == sibling_car.position():
                return False

        must_stop = car.position() == stop_position and not self.traffic_light_service.is_passable(car.lane_index)
        return not must_stop

    def has_collision(self):
        for car in self.cars:
            if car.collides_with_any(self.cars):
                return True

        return False

    def __remove_dead_cars(self):
        # TODO: This removes cars which had to be spawned outside of the screen...
        self.cars = [car for car in self.cars if car.is_alive()]

    def __get_lane_cars(self, lane_index):
        return [car for car in self.cars if car.lane_index == lane_index]

    def draw(self, surface: pygame.Surface):
        for car in self.cars:
            car.draw(surface)
