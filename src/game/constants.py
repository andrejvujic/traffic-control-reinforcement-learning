from src.environment.traffic.spawn_configuration import SpawnConfiguration
from src.environment.traffic.checkpoint import Checkpoint
from src.environment.traffic.vehicle_state import VehicleState

COLUMNS = 21
ROWS = 21
CELL_SIZE = 36
TRAIN_LENGTH = 10

TOTAL_LANES = 10
CAR_LANES = range(0, 8)
TRAIN_LANES = range(8, 10)

MAX_TICKS_PER_EPISODE = 1000

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
                position=(6, 12),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION)
            ),
            Checkpoint(
                position=(12, 5),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION)
            ),
            Checkpoint(
                position=(12, 12),
                callback=lambda car: (
                    car.set_direction(
                        (0, -1)
                    ),
                    car.set_angle(0.0)
                ),
            ),
            Checkpoint(
                position=(12, -1),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # West -> East, Forward
        lane_index=1,
        initial_direction=(1, 0),
        initial_angle=-90.0,
        checkpoints=[
            Checkpoint(
                position=(6, 14),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION)
            ),
            Checkpoint(
                position=(15, 14),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION)
            ),
            Checkpoint(
                position=(21, 14),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # West -> East, Right
        lane_index=1,
        initial_direction=(1, 0),
        initial_angle=-90.0,
        checkpoints=[
            Checkpoint(
                position=(6, 14),
                callback=lambda car: (
                    car.set_direction(
                        (0, 1)
                    ),
                    car.set_angle(-180.0)
                ),
            ),
            Checkpoint(
                position=(6, 14),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION)
            ),
            Checkpoint(
                position=(6, 15),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION)
            ),
            Checkpoint(
                position=(6, 21),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # South -> North, Left
        lane_index=2,
        initial_direction=(0, -1),
        initial_angle=0.0,
        checkpoints=[
            Checkpoint(
                position=(12, 8),
                callback=lambda vehicle: (
                    vehicle.set_direction(
                        (-1, 0)
                    ),
                    vehicle.set_angle(90.0)
                ),
            ),
            Checkpoint(
                position=(12, 14),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(5, 8),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(-1, 8),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # South -> North, Forward
        lane_index=3,
        initial_direction=(0, -1),
        initial_angle=0.0,
        checkpoints=[
            Checkpoint(
                position=(14, 14),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(14, 5),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(14, -1),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # South -> North, Right
        lane_index=3,
        initial_direction=(0, -1),
        initial_angle=0.0,
        checkpoints=[
            Checkpoint(
                position=(14, 14),
                callback=lambda car: (
                    car.set_direction(
                        (1, 0)
                    ),
                    car.set_angle(-90.0)
                ),
            ),
            Checkpoint(
                position=(14, 14),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(15, 14),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(21, 14),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # East -> West, Left
        lane_index=4,
        initial_direction=(-1, 0),
        initial_angle=90.0,
        checkpoints=[
            Checkpoint(
                position=(14, 8),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(8, 8),
                callback=lambda car: (
                    car.set_direction(
                        (0, 1)
                    ),
                    car.set_angle(180.0)
                ),
            ),
            Checkpoint(
                position=(8, 15),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(8, 21),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # East -> West, Forward
        lane_index=5,
        initial_direction=(-1, 0),
        initial_angle=90.0,
        checkpoints=[
            Checkpoint(
                position=(14, 6),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(5, 6),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(-1, 6),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # East -> West, Right
        lane_index=5,
        initial_direction=(-1, 0),
        initial_angle=90.0,
        checkpoints=[
            Checkpoint(
                position=(14, 6),
                callback=lambda car: (
                    car.set_direction(
                        (0, -1)
                    ),
                    car.set_angle(0.0)
                ),
            ),
            Checkpoint(
                position=(14, 6),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(14, 5),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(14, -1),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # North -> South, Left
        lane_index=6,
        initial_direction=(0, 1),
        initial_angle=180.0,
        checkpoints=[
            Checkpoint(
                position=(8, 6),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
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
            ),
            Checkpoint(
                position=(15, 12),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(21, 12),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # North -> South, Forward
        lane_index=7,
        initial_direction=(0, 1),
        initial_angle=180.0,
        checkpoints=[
            Checkpoint(
                position=(6, 6),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(6, 15),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(6, 21),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # North -> South, Right
        lane_index=7,
        initial_direction=(0, 1),
        initial_angle=180.0,
        checkpoints=[
            Checkpoint(
                position=(6, 6),
                callback=lambda car: (
                    car.set_direction(
                        (-1, 0)
                    ),
                    car.set_angle(90.0)
                ),
            ),
            Checkpoint(
                position=(6, 6),
                callback=lambda vehicle: vehicle.set_state(VehicleState.IN_INTERSECTION),
            ),
            Checkpoint(
                position=(5, 6),
                callback=lambda vehicle: vehicle.set_state(VehicleState.PASSED_INTERSECTION),
            ),
            Checkpoint(
                position=(-1, 6),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
]

TRAIN_SPAWN_CONFIGURATIONS = [
    SpawnConfiguration(  # Train, West -> East
        lane_index=8,
        initial_direction=(1, 0),
        initial_angle=0.0,
        checkpoints=[
            Checkpoint(
                position=(6, 10),
                callback=lambda train: train.set_state(VehicleState.IN_INTERSECTION)
            ),
            Checkpoint(
                position=(24, 10),
                callback=lambda train: train.set_state(VehicleState.PASSED_INTERSECTION)
            ),
            Checkpoint(
                position=(30, 10),
                callback=lambda vehicle: vehicle.mark_off_screen()
            ),
        ]
    ),
    SpawnConfiguration(  # Train, South -> North
        lane_index=9,
        initial_direction=(0, -1),
        initial_angle=-90.0,
        checkpoints=[
            Checkpoint(
                position=(10, 14),
                callback=lambda train: train.set_state(VehicleState.IN_INTERSECTION)
            ),
            Checkpoint(
                position=(10, -4),
                callback=lambda train: train.set_state(VehicleState.PASSED_INTERSECTION)
            ),
            Checkpoint(
                position=(10, -10),
                callback=lambda vehicle: vehicle.mark_off_screen()
            )
        ]
    ),
]

TRAFFIC_LIGHT_PHASES = [
    {0, 1},      # West -> East Left + Forward/Right
    {2, 3},      # South -> North Left + Forward/Right
    {4, 5},      # East -> West Left + Forward/Right
    {6, 7},      # North -> South Left + Forward/Right

    {1, 5},      # West -> East Forward/Right + East -> West Forward/Right
    {3, 7},      # South -> North Forward/Right + North -> South Forward/Right

    {1, 6},      # West -> East Forward/Right + North -> South Left
    {3, 0},      # South -> North Forward/Right + West -> East Left
    {5, 2},      # East -> West Forward/Right + South -> North Left
    {7, 4},      # North -> South Forward/Right + East -> West Left

    {8},
    {9},
]

CAR_MOVEMENT_INTERVAL = 1
CAR_SPAWN_INTERVAL = 3
CAR_SPAWN_PROBABILITY = 0.75

TRAIN_MOVEMENT_INTERVAL = 1
TRAIN_SPAWN_INTERVAL = 64
TRAIN_SPAWN_PROBABILITY = 0.5
