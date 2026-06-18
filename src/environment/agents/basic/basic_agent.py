from src.game.constants import TOTAL_LANES, CAR_LANES, TRAIN_LANES


class BasicAgent:
    def __init__(self):
        self.ticks = 0
        self.active_lane_index = 0

        self.previous_action = self.__default_action()

    def __default_action(self):
        return [False for _ in range(TOTAL_LANES)]

    def __rotate_active_lane(self):
        self.active_lane_index = self.active_lane_index + 2
        if self.active_lane_index >= TOTAL_LANES - 2:
            self.active_lane_index = 0

    def __car_action(self):
        next_action = self.__default_action()
        next_action[self.active_lane_index] = True
        next_action[self.active_lane_index + 1] = True

        self.__rotate_active_lane()

        return next_action

    def __train_action_if_required(self, approaching_count_per_lane):
        for lane_index in TRAIN_LANES:
            if approaching_count_per_lane[lane_index] > 0:
                next_action = self.__default_action()
                next_action[lane_index] = True
                return next_action

    def __next_action(self, approaching_count_per_lane):
        next_action = self.__train_action_if_required(approaching_count_per_lane)
        if next_action:
            return next_action

        return self.__car_action()

    def next_action(self, environment_state):
        self.ticks = self.ticks + 1

        approaching_count_per_lane = environment_state[:TOTAL_LANES]
        in_intersection_count_per_lane = environment_state[TOTAL_LANES:]
        anyone_in_intersection = any([vehicle_count > 0 for vehicle_count in in_intersection_count_per_lane])

        if self.ticks >= self.UPDATE_INTERVAL:
            if anyone_in_intersection:
                return self.__default_action()

            self.ticks = 0

            next_action = self.__next_action(approaching_count_per_lane)
            self.previous_action = next_action
            return next_action

        return self.previous_action

    UPDATE_INTERVAL = 35
