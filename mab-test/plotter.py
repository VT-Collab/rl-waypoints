import numpy as np
import pickle
import matplotlib.pyplot as plt



h1 = pickle.load(open("runs/high_rewards/rewards1.pkl", "rb"))
h2 = pickle.load(open("runs/high_rewards/rewards2.pkl", "rb"))
h3 = pickle.load(open("runs/high_rewards/rewards3.pkl", "rb"))
h4 = pickle.load(open("runs/high_rewards/rewards4.pkl", "rb"))
h5 = pickle.load(open("runs/high_rewards/rewards5.pkl", "rb"))
H = np.array([h1, h2, h3, h4, h5])
meanH = np.mean(H, axis=0)

l1 = pickle.load(open("runs/low_rewards/rewards1.pkl", "rb"))
l2 = pickle.load(open("runs/low_rewards/rewards2.pkl", "rb"))
l3 = pickle.load(open("runs/low_rewards/rewards3.pkl", "rb"))
l4 = pickle.load(open("runs/low_rewards/rewards4.pkl", "rb"))
l5 = pickle.load(open("runs/low_rewards/rewards5.pkl", "rb"))
L = np.array([l1, l2, l3, l4, l5])
meanL = np.mean(L, axis=0)

plt.plot(meanH, 'b.-')
plt.plot(meanL, 'ro-')
plt.show()