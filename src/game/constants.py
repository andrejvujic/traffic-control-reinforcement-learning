from src.environment.traffic.spawn_configuration import SpawnConfiguration
from src.environment.traffic.checkpoint import Checkpoint

COLUMNS = 21
ROWS = 21
CELL_SIZE = 36
TRAIN_LENGTH = 10

SCREEN_WIDTH = CELL_SIZE * COLUMNS
SCREEN_HEIGHT = CELL_SIZE * ROWS

TRAFFIC_LIGHT_POSITIONS = [
    (4, 16),   # West -> East, Left
    (4, 17),   # West -> East, Forward/Right

    (16, 16),  # South -> North, Left
    (17, 16),  # South -> North, Forward/Right

    (16, 4),   # East -> West, Left
    (16, 3),   # East -> West, Forward/Right

    (4, 4),    # North -> South, Left
    (3, 4),    # North -> South, Forward/Right

    (5, 9),    # Train, West -> East
    (9, 15),   # Train, South -> North
]

VEHICLE_SPAWN_POSITIONS = [
    (0, 12),
    (0, 14),
    (12, 21),
    (14, 21),
    (21, 8),
    (21, 6),
    (8, 0),
    (6, 0),
    (0, 10),
    (10, 20)
]

VEHICLE_STOP_POSITIONS = [
    (5, 12),
    (5, 14),
    (12, 15),
    (14, 15),
    (15, 8),
    (15, 6),
    (8, 5),
    (6, 5),
    (5, 10),
    (10, 15)
]

CAR_SPAWN_CONFIGURATIONS = [
    SpawnConfiguration(  # West -> East, Left
        lane_index=0,
        initial_direction=(1, 0),
        initial_angle=-90.0,
        checkpoints=[
            Checkpoint(
                position=(12, 12),
                callback=lambda car: (
                    car.set_direction(
                        (0, -1)
                    ),
                    car.set_angle(
                        0.0
                    )
                ),
            )
        ]
    ),
    SpawnConfiguration(  # West -> East, Forward/Right
        lane_index=1,
        initial_direction=(1, 0),
        initial_angle=-90.0,
        checkpoints=[]
    ),
    SpawnConfiguration(  # South -> North, Left
        lane_index=2,
        initial_direction=(0, -1),
        initial_angle=0.0,
        checkpoints=[
            Checkpoint(
                position=(12, 8),
                callback=lambda car: (
                    car.set_direction(
                        (-1, 0)
                    ),
                    car.set_angle(
                        90.0
                    )
                ),
            )
        ]
    ),
    SpawnConfiguration(  # South -> North, Forward/Right
        lane_index=3,
        initial_direction=(0, -1),
        initial_angle=0.0,
        checkpoints=[]
    ),
    SpawnConfiguration(  # East -> West, Left
        lane_index=4,
        initial_direction=(-1, 0),
        initial_angle=90.0,
        checkpoints=[
            Checkpoint(
                position=(8, 8),
                callback=lambda car: (
                    car.set_direction(
                        (0, 1)
                    ),
                    car.set_angle(
                        180.0
                    )
                ),
            )
        ]
    ),
    SpawnConfiguration(  # East -> West, Forward/Right
        lane_index=5,
        initial_direction=(-1, 0),
        initial_angle=90.0,
        checkpoints=[]
    ),
    SpawnConfiguration(  # North -> South, Left
        lane_index=6,
        initial_direction=(0, 1),
        initial_angle=180.0,
        checkpoints=[
            Checkpoint(
                position=(8, 12),
                callback=lambda car: (
                    car.set_direction(
                        (1, 0)
                    ),
                    car.set_angle(
                        -90.0
                    )
                ),
            )
        ]
    ),
    SpawnConfiguration(  # North -> South, Forward/Right
        lane_index=7,
        initial_direction=(0, 1),
        initial_angle=180.0,
        checkpoints=[]
    ),
]

TRAIN_SPAWN_CONFIGRUATIONS = [
    SpawnConfiguration(  # West -> East
        lane_index=8,
        initial_direction=(1, 0),
        initial_angle=0.0,
        checkpoints=[]
    ),
    SpawnConfiguration(  # South -> North
        lane_index=9,
        initial_direction=(0, -1),
        initial_angle=-90.0,
        checkpoints=[]
    ),
]


CAR_MOVEMENT_INTERVAL = 4
CAR_SPAWN_INTERVAL = 12
CAR_SPAWN_PROBABILITY = 0.75

TRAIN_MOVEMENT_INTERVAL = 5
TRAIN_SPAWN_INTERVAL = 24
TRAIN_SPAWN_PROBABILITY = 0.5
