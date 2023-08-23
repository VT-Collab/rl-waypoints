import numpy as np
import random


class MyMemory:
    def __init__(self):
        self.buffer = []
        self.position = 0

    # push one datapoint
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position += 1

    # sample a random batch of datapoints
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    # size of dataset
    def __len__(self):
        return len(self.buffer)
