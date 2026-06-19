from collections import deque
import torch as T
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        self.batch_size = batch_size

        self.states = deque([], memory_size)
        self.new_states = deque([], memory_size)
        self.actions = deque([], memory_size)
        self.rewards = deque([], memory_size)
        self.terminated_flags = deque([], memory_size)

    def add(self, state, action, new_state, reward, terminated_flag):
        self.states.append(state)
        self.actions.append(action)
        self.new_states.append(new_state)
        self.rewards.append(reward)
        self.terminated_flags.append(terminated_flag)

    def sample(self):
        return DataLoader(
            TensorDataset(
                T.tensor(self.states, dtype=T.float32),
                T.tensor(self.actions, dtype=T.int32),
                T.tensor(self.new_states, dtype=T.float32),
                T.tensor(self.rewards, dtype=T.float32),
                T.tensor(self.terminated_flags, dtype=T.float32)
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
