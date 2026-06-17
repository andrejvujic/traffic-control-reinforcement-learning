from src.environment.traffic.vehicle_spawner import VehicleSpawner
from src.environment.traffic.car import Car
from src.game.constants import CAR_SPAWN_POSITIONS, CAR_SPAWN_CONFIGURATIONS, CELL_SIZE

import pygame
import random
import os


class CarSpawner(VehicleSpawner):
    def __init__(self, assets_directory):
        super().__init__(assets_directory)

    def spawn(self, existing_cars: list[Car]):
        spawn_configuration = self.get_random_spawn_configuration()

        return Car(
            texture=self.get_random_texture(),
            initial_position=self.generate_spawn_position(
                spawn_configuration,
                existing_cars
            ),
            **spawn_configuration.as_dict()
        )

    def generate_spawn_position(self, spawn_configuration, existing_cars):
        default_position = CAR_SPAWN_POSITIONS[spawn_configuration.lane_index]
        sibling_cars = [
            sibling_car for sibling_car in existing_cars
            if sibling_car.lane_index == spawn_configuration.lane_index
        ]

        for sibling_car in sibling_cars:
            if sibling_car.position() == default_position:
                dx, dy = spawn_configuration.initial_direction
                last_sibling = sibling_cars[-1]
                sibling_x, sibling_y = last_sibling.position()
                return sibling_x - dx, sibling_y - dy

        return default_position

    def load_textures(self):
        textures = []

        for file_name in os.listdir(self.assets_directory):
            _, file_extension = file_name.split('.')
            if file_extension not in ['bmp']:
                continue

            texture = pygame.image.load(
                os.path.join(self.assets_directory, file_name)
            ).convert_alpha()

            textures.append(
                pygame.transform.scale(
                    texture,
                    (CELL_SIZE, CELL_SIZE)
                )
            )

        return textures

    def get_random_texture(self):
        return random.choice(self.textures)

    def get_random_spawn_configuration(self):
        return random.choice(CAR_SPAWN_CONFIGURATIONS)
