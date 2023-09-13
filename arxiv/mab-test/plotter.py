import numpy as np
import pickle
import matplotlib.pyplot as plt



h1 = pickle.load(open("runs/rewards/high.pkl", "rb"))
l1 = pickle.load(open("runs/rewards/low.pkl", "rb"))

plt.plot(h1, 'b.-')
plt.plot(l1, 'ro-')
plt.show()