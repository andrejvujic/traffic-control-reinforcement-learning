from src.game.constants import CELL_SIZE
import pygame
import os
import csv


class Map:
    def __init__(self):
        self.__map = self.__load_map()
        self.__tiles = self.__load_tiles()

    def draw(self, surface: pygame.Surface):
        for row_index, row in enumerate(self.__map):
            for column_index, column in enumerate(row):
                scaled_tile = pygame.transform.scale(
                    self.__tiles[column],
                    (CELL_SIZE, CELL_SIZE)
                )

                surface.blit(
                    scaled_tile,
                    (CELL_SIZE * column_index, CELL_SIZE * row_index)
                )

    def __load_tiles(self):
        assets = {}

        for file_name in os.listdir(self.TILES_DIRECTORY):
            asset_name, file_extension = file_name.split('.')
            if file_extension not in ['bmp']:
                continue

            file_path = os.path.join(self.TILES_DIRECTORY, file_name)
            assets[asset_name] = pygame.image.load(file_path).convert()

        return assets

    def __load_map(self):
        with open(self.MAP_FILE_PATH, 'r') as f:
            lines = f.readlines()
            return [row for row in csv.reader(lines)]

    TILES_DIRECTORY = 'assets/tiles'
    MAP_FILE_PATH = 'world.csv'
