from src.game.constants import VEHICLE_SPAWN_POSITIONS, TRAIN_SPAWN_CONFIGURATIONS, CELL_SIZE, TRAIN_LENGTH
from src.environment.traffic.train import Train
from src.environment.traffic.vehicle_spawner import VehicleSpawner
import random
import pygame
import os


class TrainSpawner(VehicleSpawner):
    def __init__(self, assets_directory):
        self.assets_directory = assets_directory
        self.textures = self.load_textures()

    def spawn(self, existing_trains):
        spawn_configuration = self.get_random_spawn_configuration()
        if any([existing_train.lane_index == spawn_configuration.lane_index for existing_train in existing_trains]):
            return

        return Train(
            texture=self.get_random_texture(),
            initial_position=self.generate_spawn_position(spawn_configuration),
            **spawn_configuration.as_dict()
        )

    def generate_spawn_position(self, spawn_configuration):
        return VEHICLE_SPAWN_POSITIONS[
            spawn_configuration.lane_index
        ]

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
                    (CELL_SIZE * TRAIN_LENGTH, CELL_SIZE)
                )
            )

        return textures

    def get_random_spawn_configuration(self):
        return random.choice(TRAIN_SPAWN_CONFIGURATIONS)
