from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pickle
import numpy as np
import os, sys
import argparse

p = argparse.ArgumentParser()
p.add_argument('--env', type=str, default='Lift')
args = p.parse_args()

folders = ['t1', 't2', 't3', 't4', 't5']
# file_name = 'models/'+ args.env + '/' + folder + '/data.pkl'
rewards = []

for folder in folders:
    file_name = 'ours/models/'+ args.env + '/' + folder + '/data.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'][:599])

rewards = np.array(rewards)

ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
ms = savgol_filter(ms, 5, 2)
ss = savgol_filter(ss, 5, 2)

x = range(len(rewards[0]))
plt.fill_between(x, ms+ss, ms-ss, color='blue', alpha=0.2)
plt.plot(x, ms, 'b--')


folders = ['t1', 't2', 't3', 't4', 't5']
rewards = []

for folder in folders:
    file_name = 'Baselines/SAC-vanilla/models/'+ args.env + '/' + folder + '/data.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)

ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
ms = savgol_filter(ms, 5, 2)
ss = savgol_filter(ss, 5, 2)

x = range(len(rewards[0]))
plt.fill_between(x, ms+ss, ms-ss, color='black', alpha=0.2)
plt.plot(x, ms, 'k--')


folders = ['t1', 't2', 't3', 't4', 't5']
rewards = []

for folder in folders:
    file_name = 'Baselines/SAC-waypoints/models/'+ args.env + '/' + folder + '/data.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)

ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
ms = savgol_filter(ms, 5, 2)
ss = savgol_filter(ss, 5, 2)

x = range(len(rewards[0]))
plt.fill_between(x, ms+ss, ms-ss, color='red', alpha=0.2)
plt.plot(x, ms, 'r--')

folders = ['t1', 't2', 't3', 't4', 't5']
rewards = []

for folder in folders:
    file_name = 'Baselines/PPO-vanilla/models/'+ args.env + '/' + folder + '/data.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)

ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
ms = savgol_filter(ms, 5, 2)
ss = savgol_filter(ss, 5, 2)

x = range(len(rewards[0]))
plt.fill_between(x, ms+ss, ms-ss, color='green', alpha=0.2)
plt.plot(x, ms, 'g--')


folders = ['t1', 't2', 't3', 't4', 't5']
rewards = []

for folder in folders:
    file_name = 'Baselines/PPO-waypoints/models/'+ args.env + '/' + folder + '/data.pkl'
    rewards.append(pickle.load(open(file_name, 'rb'))['reward'])

rewards = np.array(rewards)

ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
ms = savgol_filter(ms, 5, 2)
ss = savgol_filter(ss, 5, 2)

x = range(len(rewards[0]))
plt.fill_between(x, ms+ss, ms-ss, color='cyan', alpha=0.2)
plt.plot(x, ms, 'c--')

plt.title(args.env)
plt.ylabel('reward')
plt.xlabel('episodes')

# plt.savefig(args.env + '.png')
plt.show()




# rewards = []

# for root, dirs, files in os.walk('models/ours/Lift'):
#     print(root)
#     # for folder in os.listdir(root):
#     for file in os.listdir(root):
#         if file.endswith('.pkl'):
#             print(file)
#             data = pickle.load(open(root+'/' + file, 'rb'))['reward'][:599]
#             rewards.append(np.array(data))


# rewards = np.array(rewards)

# ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
# ms = savgol_filter(ms, 20, 2)
# ss = savgol_filter(ss, 20, 2)

# x = range(599)
# plt.fill_between(x, ms+ss, ms-ss, color='blue', alpha=0.2)
# plt.plot(x, ms, 'b--')


# plt.show()