from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pickle
import numpy as np
import os, sys

# o1 = np.array(pickle.load(open("models/Lift/1/data.pkl", "rb"))['reward'])
# o2 = np.array(pickle.load(open("models/Lift/2/data.pkl", "rb"))['reward'])
# # o3 = np.array(pickle.load(open("models/Lift/3/data.pkl", "rb"))['reward'])
# o4 = np.array(pickle.load(open("models/Lift/4/data.pkl", "rb"))['reward'])
# o5 = np.array(pickle.load(open("models/Lift/5/data.pkl", "rb"))['reward'])

# # s1 = pickle.load(open("sac/wipe/rewards1.pkl", "rb"))
# # s2 = pickle.load(open("sac/wipe/rewards2.pkl", "rb"))
# # s3 = pickle.load(open("sac/wipe/rewards3.pkl", "rb"))
# # s4 = pickle.load(open("sac/wipe/rewards4.pkl", "rb"))
# # s5 = pickle.load(open("sac/wipe/rewards5.pkl", "rb"))

# ours = np.array([o1, o2, o4, o5])
# # sac = np.array([s1, s2, s3, s4, s5])

# mo, so = np.mean(ours, axis=0), np.std(ours, axis=0) / np.sqrt(4)
# # ms, ss = np.mean(sac, axis=0), np.std(sac, axis=0) / np.sqrt(5)
# mo = savgol_filter(mo, 10, 2)
# # ms = savgol_filter(ms, 10, 2)

# x = range(500)
# # plt.fill_between(x, ms+ss, ms-ss, color='black', alpha=0.2)
# plt.fill_between(x, mo+so, mo-so, alpha=0.2)
# # plt.plot(x, ms, 'k--')
# plt.plot(x, mo, 'b-')
# plt.show()

rewards = []

for root, dirs, files in os.walk('models/sac/Lift/'):
    print(root)
    # for folder in os.listdir(root):
    for file in os.listdir(root):
        if file.endswith('.pkl'):
            rewards.append(np.array(pickle.load(open(root+'/' + file, 'rb'))))


rewards = np.array(rewards)

ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
ms = savgol_filter(ms, 20, 2)
ss = savgol_filter(ss, 20, 2)

x = range(500)
plt.fill_between(x, ms+ss, ms-ss, color='black', alpha=0.2)
plt.plot(x, ms, 'k--')



rewards = []

for root, dirs, files in os.walk('models/ours/Lift'):
    print(root)
    # for folder in os.listdir(root):
    for file in os.listdir(root):
        if file.endswith('.pkl'):
            print(file)
            data = pickle.load(open(root+'/' + file, 'rb'))['reward'][:599]
            rewards.append(np.array(data))


rewards = np.array(rewards)

ms, ss = np.mean(rewards, axis=0), np.std(rewards, axis=0)/np.sqrt(5)
ms = savgol_filter(ms, 20, 2)
ss = savgol_filter(ss, 20, 2)

x = range(599)
plt.fill_between(x, ms+ss, ms-ss, color='blue', alpha=0.2)
plt.plot(x, ms, 'b--')


plt.show()