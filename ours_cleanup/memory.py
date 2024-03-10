import numpy as np
import random
import torch

class MyMemory:
    def __init__(self):
        self.buffer = []
        self.position = 0

    # push trajectory, reward pair
    def push(self, traj, reward):
        self.buffer.append(None)
        self.buffer[self.position] = (traj, reward)
        self.position += 1

        # self.buffer.append(None)
        # self.buffer[-1] = (traj, reward)
        # while len(self.buffer) >= 100:
        #     self.buffer.pop(0)
        # self.position += 1

    # sample a random batch of trajs, rewards
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        trajs, rewards = map(np.stack, zip(*batch))
        return trajs, rewards

    # size of dataset
    def __len__(self):
        return len(self.buffer)        

"""FOR AE"""
# class MyMemory:
#     def __init__(self):
#         self.buffer = []
#         self.position = 0

#     # push trajectory, reward pair
#     def push(self, objs, wp, reward):
#         self.buffer.append(None)
#         self.buffer[self.position] = (objs, wp, reward)
#         self.position += 1

#         # self.buffer.append(None)
#         # self.buffer[-1] = (traj, reward)
#         # while len(self.buffer) >= 100:
#         #     self.buffer.pop(0)
#         # self.position += 1

#     # sample a random batch of trajs, rewards
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         start, wp, rewards = map(np.stack, zip(*batch))
#         return start, wp, rewards 

#     # size of dataset
#     def __len__(self):
#         return len(self.buffer)        


# import numpy as np
# import random


# class MyMemory:
#     def __init__(self):
#         self.buffer = []
#         self.position = 0

#     # push trajectory, reward pair
#     def push(self, traj, objs, reward):
#         self.buffer.append(None)
#         self.buffer[self.position] = (traj, objs, reward)
#         self.position += 1

#         # self.buffer.append(None)
#         # self.buffer[-1] = (traj, reward)
#         # while len(self.buffer) >= 100:
#         #     self.buffer.pop(0)
#         # self.position += 1

#     # sample a random batch of trajs, rewards
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         trajs, objs, rewards = map(np.stack, zip(*batch))
#         return trajs, objs, rewards

#     # size of dataset
#     def __len__(self):
#         return len(self.buffer)        