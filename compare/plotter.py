from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pickle
import numpy as np

o1 = pickle.load(open("ours/lift2/rewards1.pkl", "rb"))
o2 = pickle.load(open("ours/lift2/rewards2.pkl", "rb"))
o3 = pickle.load(open("ours/lift2/rewards3.pkl", "rb"))
o4 = pickle.load(open("ours/lift2/rewards4.pkl", "rb"))
o5 = pickle.load(open("ours/lift2/rewards5.pkl", "rb"))

s1 = pickle.load(open("sac/lift/rewards1.pkl", "rb"))
s2 = pickle.load(open("sac/lift/rewards2.pkl", "rb"))
s3 = pickle.load(open("sac/lift/rewards3.pkl", "rb"))
s4 = pickle.load(open("sac/lift/rewards4.pkl", "rb"))
s5 = pickle.load(open("sac/lift/rewards5.pkl", "rb"))
s6 = pickle.load(open("sac/lift/rewards6.pkl", "rb"))
s7 = pickle.load(open("sac/lift/rewards7.pkl", "rb"))
s8 = pickle.load(open("sac/lift/rewards8.pkl", "rb"))
s9 = pickle.load(open("sac/lift/rewards9.pkl", "rb"))
s10 = pickle.load(open("sac/lift/rewards10.pkl", "rb"))

ours = np.array([o1, o2, o3, o4, o5])
sac = np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])

mo, so = np.mean(ours, axis=0), np.std(ours, axis=0) / np.sqrt(len(ours))
ms, ss = np.mean(sac, axis=0), np.std(sac, axis=0) / np.sqrt(len(sac))
mo = savgol_filter(mo, 10, 2)
ms = savgol_filter(ms, 10, 2)

x = range(500)
plt.fill_between(x, ms+ss, ms-ss, color='black', alpha=0.2)
plt.fill_between(x, mo+so, mo-so, alpha=0.2)
plt.plot(x, ms, 'k--')
plt.plot(x, mo, 'b-')
plt.show()


# collab@192.168.1.18
# collab@172.29.29.124