from src.environment.traffic.car import Car
from src.environment.traffic.traffic_light_service import TrafficLightService
from src.game.constants import CAR_MOVEMENT_INTERVAL, CAR_SPAWN_INTERVAL, CAR_SPAWN_PROBABILITY, CAR_CONFIGURATIONS, CAR_SPAWN_POSITIONS, CAR_STOP_POSITIONS, CELL_SIZE
from src.game.utilities import random_bool

import os
import random
import pygame


class VehicleService:
    def __init__(self, traffic_light_service: TrafficLightService):
        self.traffic_light_service = traffic_light_service
        self.ticks = 0
        self.cars: list[Car] = []
        self.__car_textures = self.__load_car_textures()

    def update(self):
        self.ticks = self.ticks + 1

        if self.ticks % CAR_SPAWN_INTERVAL == 0 and random_bool(CAR_SPAWN_PROBABILITY):
            configuration = self.__random_car_configuration()

            self.cars.append(
                Car(
                    texture=self.__random_car_texture(),
                    initial_position=self.__car_spawn_position(
                        configuration.lane_index,
                        configuration.initial_direction
                    ),
                    **configuration.as_dict()
                )
            )

        for car in self.cars:
            if self.ticks % CAR_MOVEMENT_INTERVAL == 0 and self.can_move(car):
                car.move()

    def can_move(self, car: Car):
        stop_position = CAR_STOP_POSITIONS[car.lane_index]
        sibling_cars = self.__lane_cars(car.lane_index)

        for sibling_car in sibling_cars:
            if car.next_position() == sibling_car.position():
                return False

        must_stop = car.position() == stop_position and not self.traffic_light_service.is_enabled(car.lane_index)
        return not must_stop

    def __lane_cars(self, lane_index):
        return [car for car in self.cars if car.lane_index == lane_index]

    def __car_spawn_position(self, lane_index, movement_direction):
        default_position = CAR_SPAWN_POSITIONS[lane_index]
        sibling_cars = self.__lane_cars(lane_index)

        for sibling_car in sibling_cars:
            if sibling_car.position() == default_position:
                dx, dy = movement_direction
                last_sibling = sibling_cars[-1]
                sibling_x, sibling_y = last_sibling.position()
                return sibling_x - dx, sibling_y - dy

        return default_position

    def draw(self, surface: pygame.Surface):
        for car in self.cars:
            car.draw(surface)

    def __load_car_textures(self):
        textures = []

        for file_name in os.listdir(self.CAR_TEXTURES_DIRECTORY):
            _, file_extension = file_name.split('.')
            if file_extension not in ['bmp']:
                continue

            texture = pygame.image.load(
                os.path.join(self.CAR_TEXTURES_DIRECTORY, file_name)
            ).convert_alpha()

            textures.append(
                pygame.transform.scale(
                    texture,
                    (CELL_SIZE, CELL_SIZE)
                )
            )

        return textures

    def __random_car_texture(self):
        return random.choice(self.__car_textures)

    def __random_car_configuration(self):
        return random.choice(CAR_CONFIGURATIONS)

    CAR_TEXTURES_DIRECTORY = 'assets/cars'
