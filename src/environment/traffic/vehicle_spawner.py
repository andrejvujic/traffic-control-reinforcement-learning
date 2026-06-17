from abc import ABC, abstractmethod
import random


class VehicleSpawner(ABC):
    def __init__(self, assets_directory):
        self.assets_directory = assets_directory
        self.textures = self.load_textures()

    @abstractmethod
    def spawn(self):
        pass

    @abstractmethod
    def generate_spawn_position(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_textures(self):
        pass

    def get_random_texture(self):
        return random.choice(self.textures)

    @abstractmethod
    def get_random_spawn_configuration(self):
        pass
