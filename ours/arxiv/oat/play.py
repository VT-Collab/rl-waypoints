import numpy as np
import torch
from memory import MyMemory
from method import Method
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
from scipy.spatial.transform import Rotation as R

# # training parameters
# batch_size = 30

# # load default controller parameters for Operational Space Control (OSC)
# controller_config = load_controller_config(default_controller="OSC_POSE")

# # create environment instance
# env = suite.make(
#     env_name="Stack", # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     controller_configs=controller_config,
#     has_renderer=True,
#     reward_shaping=True,
#     control_freq=10,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     initialization_noise=None,
#     use_latch=True,
# )

# # Main loop
# total_steps = 0

# # initialize variables
# episode_reward = 0
# done, truncated = False, False
# obs = env.reset()

# lin = obs['robot0_eef_pos']
# quat = obs['robot0_eef_quat']
# r = R.from_quat(quat)
# ang_eul = r.as_euler('xyz', degrees=False)
# pos = np.concatenate((lin, ang_eul))

# # select optimal trajectory
# traj = pickle.load(open('models/Door_ang3/traj_64.2.pkl', 'rb'))
# traj_mat = np.reshape(traj, (2, 6)) + pos
# gripper_mat = np.reshape(traj, (2, 6)) + pos

# flag = True

# for timestep in range(100):

#     env.render()    # toggle this when we don't want to render
#     if flag:
#         print("PRESS ENTER TO START")
#         # input()
#         flag = False

#     # convert traj to actions
#     state = pos
#     g_state = obs['robot0_gripper_qpos']
#     widx = int(np.floor(timestep / 50))
#     error = traj_mat[widx, :] - state
#     # error_g = gripper_mat[widx, :] - g_state
#     full_action = np.array(list(10. * error) + [-1])

#     # take step
#     obs, reward, done, _ = env.step(full_action)
#     episode_reward += reward
#     total_steps += 1

# print(episode_reward)



# # training parameters
# batch_size = 30

# # load default controller parameters for Operational Space Control (OSC)
# controller_config = load_controller_config(default_controller="OSC_POSE")

# # create environment instance
# env = suite.make(
#     env_name="Stack", # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     controller_configs=controller_config,
#     has_renderer=True,
#     reward_shaping=True,
#     control_freq=10,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     initialization_noise=None,
#     # use_latch=False,
# )

# # Main loop
# total_steps = 0

# # initialize variables
# episode_reward = 0
# done, truncated = False, False
# obs = env.reset()

# # select optimal trajectory
# traj = pickle.load(open('models/Stack1/traj_53.93.pkl', 'rb'))
# traj_mat = np.reshape(traj, (4,5))[:, :3] + obs['robot0_eef_pos']
# gripper_mat = np.reshape(traj, (4, 5))[:, 3:] + obs['robot0_gripper_qpos']

# flag = True

# for timestep in range(100):

#     env.render()    # toggle this when we don't want to render
#     if flag:
#         print("PRESS ENTER TO START")
#         # input()
#         flag = False

#     # convert traj to actions
#     state = obs['robot0_eef_pos']
#     g_state = obs['robot0_gripper_qpos']
#     widx = int(np.floor(timestep / 25))
#     error = traj_mat[widx, :] - state
#     error_g = gripper_mat[widx, :] - g_state
#     full_action = np.array(list(10. * error) + [0.]*3 + list([-1 if error_g[0] > 0 else 1]))

#     # take step
#     obs, reward, done, _ = env.step(full_action)
#     episode_reward += reward
#     total_steps += 1

# print(episode_reward)





# training parameters
batch_size = 30

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    controller_configs=controller_config,
    has_renderer=True,
    reward_shaping=True,
    control_freq=10,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    initialization_noise=None,
    # use_latch=False,
)
# while True:
# Main loop
total_steps = 0
# initialize variables
episode_reward = 0
done, truncated = False, False
obs = env.reset()

# select optimal trajectory
traj = pickle.load(open('models/Lift/ae_test/traj.pkl', 'rb'))


traj_mat = np.reshape(traj, (3,4))[:, :3] + obs['robot0_eef_pos']
gripper_mat = np.reshape(traj, (3, 4))[:, 3:]

print(traj_mat)
# print(obs['Bread_pos'])
# exit()
flag = True
time_s = 0
input()
for timestep in range(300):

    env.render()    # toggle this when we don't want to render
    if flag:
        print("PRESS ENTER TO START")
        # input()
        flag = False

    # convert traj to actions
    state = obs['robot0_eef_pos']
    g_state = obs['robot0_gripper_qpos']
    widx = int(np.floor(timestep / (50)))
    error = traj_mat[widx, :] - state
    # error_g = gripper_mat[widx, :] - g_state
    if time_s >= 40:
        full_action = np.array([0.]*6 + list(gripper_mat[widx]))
    else:
        full_action = np.array(list(10. * error) + [0.]*4)
    # full_action = np.array(list(10. * error) + [0.]*3 + list(gripper_mat[widx]))
    time_s += 1
    if time_s >= 50:
        time_s = 1
    # take step
    obs, reward, done, _ = env.step(full_action)
    episode_reward += reward
    total_steps += 1

print(episode_reward)