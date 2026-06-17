from src.environment.traffic.car_spawn_configuration import CarSpawnConfiguration
from src.environment.traffic.car_checkpoint import CarCheckpoint

COLUMNS = 21
ROWS = 21
CELL_SIZE = 36

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

CAR_STOP_POSITIONS = [
    (5, 12),
    (5, 14),
    (12, 15),
    (14, 15),
    (15, 8),
    (15, 6),
    (8, 5),
    (6, 5)
]

CAR_SPAWN_POSITIONS = [
    (0, 12),
    (0, 14),
    (12, 21),
    (14, 21),
    (21, 8),
    (21, 6),
    (8, 0),
    (6, 0)
]

CAR_SPAWN_CONFIGURATIONS = [
    CarSpawnConfiguration(  # West -> East, Left
        lane_index=0,
        initial_direction=(1, 0),
        initial_angle=-90.0,
        checkpoints=[
            CarCheckpoint(
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
    CarSpawnConfiguration(  # West -> East, Forward/Right
        lane_index=1,
        initial_direction=(1, 0),
        initial_angle=-90.0,
        checkpoints=[]
    ),
    CarSpawnConfiguration(  # South -> North, Left
        lane_index=2,
        initial_direction=(0, -1),
        initial_angle=0.0,
        checkpoints=[
            CarCheckpoint(
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
    CarSpawnConfiguration(  # South -> North, Forward/Right
        lane_index=3,
        initial_direction=(0, -1),
        initial_angle=0.0,
        checkpoints=[]
    ),
    CarSpawnConfiguration(  # East -> West, Left
        lane_index=4,
        initial_direction=(-1, 0),
        initial_angle=90.0,
        checkpoints=[
            CarCheckpoint(
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
    CarSpawnConfiguration(  # East -> West, Forward/Right
        lane_index=5,
        initial_direction=(-1, 0),
        initial_angle=90.0,
        checkpoints=[]
    ),
    CarSpawnConfiguration(  # North -> South, Left
        lane_index=6,
        initial_direction=(0, 1),
        initial_angle=180.0,
        checkpoints=[
            CarCheckpoint(
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
    CarSpawnConfiguration(  # North -> South, Forward/Right
        lane_index=7,
        initial_direction=(0, 1),
        initial_angle=180.0,
        checkpoints=[]
    ),
]

CAR_MOVEMENT_INTERVAL = 4
CAR_SPAWN_INTERVAL = 12
CAR_SPAWN_PROBABILITY = 0.75
