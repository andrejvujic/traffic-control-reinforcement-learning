from src.game.constants import TOTAL_LANE_COUNT, CAR_LANES, TRAIN_LANES, TRAFFIC_LIGHT_PHASES
from src.environment.agents.agent import Agent


class BasicAgent(Agent):
    def __init__(self, update_interval=48):
        super().__init__(
            name='Basic'
        )
        self.update_interval = update_interval

        self.ticks = 0
        self.active_lane_index = 0

        self.previous_action = self.__all_red_action()

    def next_action(self, state):
        self.ticks = self.ticks + 1

        approaching_count_per_lane = self.__approaching_count_per_lane(state)
        passing_per_lane = self.__passing_per_lane(state)
        anyone_in_intersection = any([vehicle_count > 0 for vehicle_count in passing_per_lane])

        if self.ticks >= self.update_interval:
            if anyone_in_intersection:
                return self.__all_red_action()

            self.ticks = 0

            next_action = self.__next_action(approaching_count_per_lane)
            self.previous_action = next_action
            return next_action

        return self.previous_action

    def __all_red_action(self):
        return len(TRAFFIC_LIGHT_PHASES)

    def __phase_index_for_lanes(self, lane_indexes):
        return TRAFFIC_LIGHT_PHASES.index(set(lane_indexes))

    def __rotate_active_lane(self):
        self.active_lane_index = self.active_lane_index + 2
        if self.active_lane_index >= TOTAL_LANE_COUNT - 2:
            self.active_lane_index = 0

    def __car_action(self):
        next_action = self.__phase_index_for_lanes([
            self.active_lane_index,
            self.active_lane_index + 1
        ])

        self.__rotate_active_lane()

        return next_action

    def __train_action_if_required(self, approaching_count_per_lane):
        for lane_index in TRAIN_LANES:
            if approaching_count_per_lane[lane_index] > 0:
                return self.__phase_index_for_lanes([lane_index])

    def __next_action(self, approaching_count_per_lane):
        next_action = self.__train_action_if_required(approaching_count_per_lane)
        if next_action is not None:
            return next_action

        return self.__car_action()

    def __approaching_count_per_lane(self, state):
        car_approaching_start = TOTAL_LANE_COUNT
        train_approaching_start = car_approaching_start + len(CAR_LANES)

        approaching_count_per_lane = [0 for _ in range(TOTAL_LANE_COUNT)]

        for offset, lane_index in enumerate(CAR_LANES):
            approaching_count_per_lane[lane_index] = state[car_approaching_start + offset]

        for offset, lane_index in enumerate(TRAIN_LANES):
            approaching_count_per_lane[lane_index] = state[train_approaching_start + offset]

        return approaching_count_per_lane

    def __passing_per_lane(self, state):
        passing_start = TOTAL_LANE_COUNT + len(CAR_LANES) + len(TRAIN_LANES)
        return state[passing_start:passing_start + TOTAL_LANE_COUNT]
