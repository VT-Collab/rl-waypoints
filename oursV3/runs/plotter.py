from matplotlib import pyplot as plt
import pickle
import numpy as np

xi_robot, xi_cube = pickle.load(open("ours_lift_12-48/traj.pkl", "rb"))

xi_robot = np.array(xi_robot)
xi_cube = np.array(xi_cube)

plt.plot(xi_robot[:,1], xi_robot[:,2], 'ko')
plt.plot(xi_cube[:,1], xi_cube[:,2], 'bo')
plt.show()

plt.plot(xi_robot[:,0], xi_robot[:,2], 'ko')
plt.plot(xi_cube[:,0], xi_cube[:,2], 'bo')
plt.show()
