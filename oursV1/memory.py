import numpy as np
import random


class MyMemory:
    def __init__(self):
        self.buffer = []
        self.position = 0

    # push trajectory, reward pair
    def push(self, traj, reward):
        self.buffer.append(None)
        self.buffer[self.position] = (traj, reward)
        self.position += 1

    # sample a random batch of trajs, rewards
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        trajs, rewards = map(np.stack, zip(*batch))
        return trajs, rewards

    # size of dataset
    def __len__(self):
        return len(self.buffer)        