import numpy as np
import torch
from memory import MyMemory
from method import Method
import datetime
from torch.utils.tensorboard import SummaryWriter
from scipy import interpolate
import pickle


# training parameters
batch_size = 30
dim_low = 3
dim_full = 20

# Agent
agent = Method(traj_dim=dim_low, state_dim=1)

# Memory
memory = MyMemory()

# Logger
run_name = 'runs/low_dim_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)
dataset = []

# Task
xi_star = 2*(np.random.rand(dim_full)-0.5)

# Main loop
total_steps = 0
for i_episode in range(1, 301):

    # initialize variables
    episode_reward = 0
    
    # get traj
    traj = 2.0*(np.random.rand(dim_low)-0.5)
    if i_episode > 40:
        traj = agent.traj_opt()

    if len(memory) > batch_size:
        for _ in range(1):
            critic_loss = agent.update_parameters(memory, batch_size)
            writer.add_scalar('model/critic', critic_loss, total_steps)

    # control robot to follow traj
    f = interpolate.interp1d(np.linspace(0, 1, dim_low), traj)
    traj_full = f(np.linspace(0, 1, dim_full))

    # execute traj
    error = np.linalg.norm(traj_full - xi_star)
    reward = -error
    episode_reward = reward
    total_steps += 1

    memory.push(traj, episode_reward)
    if episode_reward > agent.best_reward:
        agent.set_init(traj, episode_reward)
        print(episode_reward, "new best trajectory")

    writer.add_scalar('reward', episode_reward, i_episode)
    print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
    dataset.append(episode_reward)

pickle.dump(dataset, open(run_name + "/rewards.pkl", "wb"))