from src.game.constants import CELL_SIZE
import pygame
import os


class Map:
    def __init__(self):
        self.__map = self.__get_map()
        self.__tiles = self.__get_tiles()

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

    def __get_tiles(self):
        assets = {}

        for file_name in os.listdir(self.ASSETS_DIRECTORY):
            asset_name, file_extension = file_name.split('.')
            if file_extension not in ['bmp']:
                continue

            file_path = os.path.join(self.ASSETS_DIRECTORY, file_name)
            assets[asset_name] = pygame.image.load(file_path).convert()

        return assets

    def __get_map(self):
        G = self.GRASS
        R = self.ROAD
        RE = self.ROAD_EDGE
        RD = self.ROAD_DANGER
        LV = self.ROAD_LINE_VERTICAL
        LH = self.ROAD_LINE_HORIZONTAL
        T = self.TRAIN_TRACK

        EW0 = self.SIGN_EAST_WEST_0
        EW1 = self.SIGN_EAST_WEST_1

        WE0 = self.SIGN_WEST_EAST_0
        WE1 = self.SIGN_WEST_EAST_1

        NS0 = self.SIGN_NORTH_SOUTH_0
        NS1 = self.SIGN_NORTH_SOUTH_1

        SN0 = self.SIGN_SOUTH_NORTH_0
        SN1 = self.SIGN_SOUTH_NORTH_1

        T0 = self.TRACK_SUPPORT_0
        T1 = self.TRACK_SUPPORT_1
        T2 = self.TRACK_SUPPORT_2
        T3 = self.TRACK_SUPPORT_3

        TS0 = self.TRACK_STOP_0
        TS1 = self.TRACK_STOP_1

        TEW = self.TRAFFIC_LIGHT_EAST_WEST
        TWE = self.TRAFFIC_LIGHT_WEST_EAST
        TNS = self.TRAFFIC_LIGHT_NORTH_SOUTH
        TSN = self.TRAFFIC_LIGHT_SOUTH_NORTH

        TTL0 = self.TRAIN_TRAFFIC_LIGHT_0
        TTL1 = self.TRAIN_TRAFFIC_LIGHT_1

        return [
            [G, G, G, G, G, RE, R, LV, R, T1, T, T3, R, LV, R, RE, G, G, G, G, G],
            [G, G, G, G, G, RE, R, LV, R, T1, T, T3, R, LV, R, RE, G, G, G, G, G],
            [G, G, G, G, G, RE, R, LV, R, T1, T, T3, R, LV, R, RE, G, G, G, G, G],
            [G, G, G, G, G, RE, R, LV, R, T1, T, T3, R, LV, R, RE, TEW, G, G, G, G],
            [G, G, G, TNS, TNS, RE, R, LV, R, T1, T, T3, R, LV, R, RE, TEW, G, G, G, G],
            [RE, RE, RE, RE, RE, RE, NS0, LV, NS1, T1, T, T3, R, LV, R, RE, RE, RE, RE, RE, RE],

            [R, R, R, R, R, R, RD, RD, RD, RD, T, RD, RD, RD, RD, EW0, R, R, R, R, R],
            [LH, LH, LH, LH, LH, LH, RD, RD, RD, RD, T, RD, RD, RD, RD, LH, LH, LH, LH, LH, LH],
            [R, R, R, R, R, R, RD, RD, RD, RD, T, RD, RD, RD, RD, EW1, R, R, R, R, R],

            [T2, T2, T2, T2, T2, TTL1, RD, RD, RD, RD, T, RD, RD, RD, RD, T2, T2, T2, T2, T2, T2],
            [T, T, T, T, T, TS0, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T],
            [T0, T0, T0, T0, T0, T0, RD, RD, RD, RD, T, RD, RD, RD, RD, T0, T0, T0, T0, T0, T0],

            [R, R, R, R, R, WE1, RD, RD, RD, RD, T, RD, RD, RD, RD, R, R, R, R, R, R],
            [LH, LH, LH, LH, LH, LH, RD, RD, RD, RD, T, RD, RD, RD, RD, LH, LH, LH, LH, LH, LH],
            [R, R, R, R, R, WE0, RD, RD, RD, RD, T, RD, RD, RD, RD, R, R, R, R, R, R],

            [RE, RE, RE, RE, RE, RE, R, LV, R, TTL0, TS1, T3, SN1, LV, SN0, RE, RE, RE, RE, RE, RE],
            [G, G, G, G, TWE, RE, R, LV, R, T1, T, T3, R, LV, R, RE, TSN, TSN, G, G, G],
            [G, G, G, G, TWE, RE, R, LV, R, T1, T, T3, R, LV, R, RE, G, G, G, G, G],
            [G, G, G, G, G, RE, R, LV, R, T1, T, T3, R, LV, R, RE, G, G, G, G, G],
            [G, G, G, G, G, RE, R, LV, R, T1, T, T3, R, LV, R, RE, G, G, G, G, G],
            [G, G, G, G, G, RE, R, LV, R, T1, T, T3, R, LV, R, RE, G, G, G, G, G],
        ]

    ASSETS_DIRECTORY = 'assets/tiles'

    GRASS = 'grass'
    ROAD = 'road'
    ROAD_EDGE = 'road_edge'
    ROAD_DANGER = 'road_danger'
    ROAD_LINE_VERTICAL = 'road_line_vertical'
    ROAD_LINE_HORIZONTAL = 'road_line_horizontal'
    TRAIN_TRACK = 'train_track'

    SIGN_EAST_WEST_0 = 'sign_east_west_0'
    SIGN_EAST_WEST_1 = 'sign_east_west_1'

    SIGN_WEST_EAST_0 = 'sign_west_east_0'
    SIGN_WEST_EAST_1 = 'sign_west_east_1'

    SIGN_NORTH_SOUTH_0 = 'sign_north_south_0'
    SIGN_NORTH_SOUTH_1 = 'sign_north_south_1'

    SIGN_SOUTH_NORTH_0 = 'sign_south_north_0'
    SIGN_SOUTH_NORTH_1 = 'sign_south_north_1'

    TRACK_SUPPORT_0 = 'track_support_0'
    TRACK_SUPPORT_1 = 'track_support_1'
    TRACK_SUPPORT_2 = 'track_support_2'
    TRACK_SUPPORT_3 = 'track_support_3'

    TRACK_STOP_0 = 'track_stop_0'
    TRACK_STOP_1 = 'track_stop_1'

    TRAFFIC_LIGHT_EAST_WEST = 'traffic_light_east_west'
    TRAFFIC_LIGHT_WEST_EAST = 'traffic_light_west_east'
    TRAFFIC_LIGHT_NORTH_SOUTH = 'traffic_light_north_south'
    TRAFFIC_LIGHT_SOUTH_NORTH = 'traffic_light_south_north'

    TRAIN_TRAFFIC_LIGHT_0 = 'train_traffic_light_0'
    TRAIN_TRAFFIC_LIGHT_1 = 'train_traffic_light_1'
