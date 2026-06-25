from torch.utils.data import DataLoader, TensorDataset
import torch as T
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.values = []
        self.end_flags = []
        self.buffer_size = 0

    def add(self, state, action, action_log_prob, reward, value, end_flag):
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.end_flags.append(end_flag)

        self.buffer_size = self.buffer_size + 1

    def clear(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.values = []
        self.end_flags = []

        self.buffer_size = 0

    def get_all(self):
        return self.states, self.actions, self.action_log_probs, self.rewards, self.values, self.end_flags

    def create_batches(self, batch_size):
        start_indices = np.arange(0, self.buffer_size, batch_size)

        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)

        return [
            np.array(indices[start_index: start_index + batch_size])
            for start_index in start_indices
        ]

    def __len__(self):
        return self.buffer_size
