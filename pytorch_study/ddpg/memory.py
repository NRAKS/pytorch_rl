import numpy as np
from collections import deque
import random


class ReplayMemory(object):
    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, item):
        self.memory.append(item)

    def extend(self, *items):
        self.memory.extend(*items)

    def clear(self):
        self.memory.clear()

    def sample(self, batch_size):
        self._assert_batch_size(batch_size)
        return random.sample(self.memory, batch_size)

    def _assert_batch_size(self, batch_size):
        assert batch_size <= self.__len__(), 'Unable to sample {} items, current buffer size {}' .format(batch_size, self.__len__())

    def __len__(self):
        return len(self.memory)

    def sample_and_split(self, batch_size):
        experiences = self.sample(batch_size)
        obs_batch = deque(maxlen=batch_size)
        act_batch = deque(maxlen=batch_size)
        new_obs_batch = deque(maxlen=batch_size)
        reward_batch = deque(maxlen=batch_size)
        terminal_batch = deque(maxlen=batch_size)

        for n in range(len(experiences)):
            obs_batch.append(experiences[n][0])
            act_batch.append(experiences[n][1])
            new_obs_batch.append(experiences[n][2])
            reward_batch.append(experiences[n][3])
            terminal_batch.append(experiences[n][4])

        # print(obs_batch)
        # print(np.array(list(obs_batch)).shape)
        obs_batch = np.array(list(obs_batch))
        act_batch = np.array(list(act_batch))
        new_obs_batch = np.array(list(new_obs_batch))
        reward_batch = np.array(list(reward_batch))
        terminal_batch = np.array(list(terminal_batch))

        return obs_batch, act_batch, new_obs_batch, reward_batch, terminal_batch