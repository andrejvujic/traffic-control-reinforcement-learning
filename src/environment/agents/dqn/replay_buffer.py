from collections import deque
import numpy as np
import torch as T


class ReplayBuffer:
    def __init__(self, memory_size):
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

    def sample(self, batch_size):
        total_experiences = len(self)
        all_indices = np.arange(0, total_experiences)
        return np.random.choice(all_indices, batch_size)

    def get_tensors(self, indices):
        states = []
        actions = []
        new_states = []
        rewards = []
        terminated_flags = []

        for index in indices:
            states.append(self.states[index])
            actions.append(self.actions[index])
            new_states.append(self.new_states[index])
            rewards.append(self.rewards[index])
            terminated_flags.append(self.terminated_flags[index])

        states = T.tensor(states, dtype=T.float32)
        actions = T.tensor(actions, dtype=T.long)
        new_states = T.tensor(new_states, dtype=T.float32)
        rewards = T.tensor(rewards, dtype=T.float32)
        terminated_flags = T.tensor(terminated_flags, dtype=T.long)

        return states, actions, new_states, rewards, terminated_flags

    def __len__(self):
        return len(self.states)
