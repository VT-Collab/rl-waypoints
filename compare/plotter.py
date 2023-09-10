from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pickle
import numpy as np

o1 = pickle.load(open("ours/rewards1.pkl", "rb"))
o2 = pickle.load(open("ours/rewards2.pkl", "rb"))
o3 = pickle.load(open("ours/rewards3.pkl", "rb"))
o4 = pickle.load(open("ours/rewards4.pkl", "rb"))
o5 = pickle.load(open("ours/rewards5.pkl", "rb"))

s1 = pickle.load(open("sac/rewards1.pkl", "rb"))
s2 = pickle.load(open("sac/rewards2.pkl", "rb"))
s3 = pickle.load(open("sac/rewards3.pkl", "rb"))
s4 = pickle.load(open("sac/rewards4.pkl", "rb"))
s5 = pickle.load(open("sac/rewards5.pkl", "rb"))

ours = np.array([o1, o2, o3, o4, o5])
sac = np.array([s1, s2, s3, s4, s5])

mo, so = np.mean(ours, axis=0), np.std(ours, axis=0) / np.sqrt(5)
ms, ss = np.mean(sac, axis=0), np.std(sac, axis=0) / np.sqrt(5)
mo = savgol_filter(mo, 10, 2)
ms = savgol_filter(ms, 10, 2)

x = range(500)
plt.fill_between(x, ms+ss, ms-ss, color='black', alpha=0.2)
plt.fill_between(x, mo+so, mo-so, alpha=0.2)
plt.plot(x, ms, 'k--')
plt.plot(x, mo, 'b-')
plt.show()