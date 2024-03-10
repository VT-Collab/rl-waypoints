from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pickle
import numpy as np
import os, sys
import argparse

p = argparse.ArgumentParser()
p.add_argument('--env', type=str, default='Lift')
args = p.parse_args()

# folders = ['test']
folders = ['milk/test', 'can/test', 'bread/test']
# file_name = 'models/'+ args.env + '/' + folder + '/data.pkl'
rewards = []

for folder in folders:
    file_name = 'ours/models/'+ args.env + '/' + folder + '/eval_reward.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)
print("ours")
print(np.mean(rewards))
print(np.std(rewards)/np.sqrt(300))



rewards = []

for folder in folders:
    file_name = 'Baselines/SAC-vanilla/models/'+ args.env + '/' + folder + '/eval_reward.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)
print("sac-vanilla")
print(np.mean(rewards))
print(np.std(rewards)/np.sqrt(300))


rewards = []

for folder in folders:
    file_name = 'Baselines/SAC-waypoints/models/'+ args.env + '/' + folder + '/eval_reward.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)
print("sac-wp")
print(np.mean(rewards))
print(np.std(rewards)/np.sqrt(300))


rewards = []

for folder in folders:
    file_name = 'Baselines/PPO-vanilla/models/'+ args.env + '/' + folder + '/eval_rewards.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)
print("ppo-vanilla")
print(np.mean(rewards))
print(np.std(rewards)/np.sqrt(300))


rewards = []

for folder in folders:
    file_name = 'Baselines/PPO-waypoints/models/'+ args.env + '/' + folder + '/eval_rewards.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)
print("ppo-wp")
print(np.mean(rewards))
print(np.std(rewards)/np.sqrt(300))




