import numpy as np
import torch
from memory import MyMemory
from method import Method
import gymnasium as gym
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle


# training parameters
batch_size = 30

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Wipe",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    reward_shaping=True,
    control_freq=10,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    initialization_noise=None,
)

# Agent
agent = Method(traj_dim=12)
agent.set_n_samples(1)

# Memory
memory = MyMemory()

# Logger
run_name = 'runs/ours_wipe_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)
reward_data = []


# Main loop
total_steps = 0
for i_episode in range(1, 501):

    # initialize variables
    episode_reward = 0
    obs = env.reset()
    robot_home = np.copy(obs['robot0_eef_pos'])

    # select optimal trajectory
    traj = 0.5*(np.random.rand(12)-0.5)
    if i_episode > 40:
        traj = agent.traj_opt()
    traj_mat = np.reshape(traj, (4,3)) + robot_home

    for widx in range(4):

        for timestep in range(25):

            # env.render()    # toggle this when we don't want to render

            if len(memory) > batch_size:
                for _ in range(1):
                    critic_loss = agent.update_parameters(memory, batch_size)
                    writer.add_scalar('model/critic', critic_loss, total_steps)

            # convert traj to actions
            state = obs['robot0_eef_pos']
            error = traj_mat[widx, :3] - state
            full_action = np.array(list(10. * error) + [0.]*3)

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward
            total_steps += 1

    memory.push(traj, episode_reward)
    if episode_reward > agent.best_reward:
        agent.set_init(traj, episode_reward)

    writer.add_scalar('reward', episode_reward, i_episode)
    print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
    reward_data.append(episode_reward)
    pickle.dump(reward_data, open(run_name + "/rewards.pkl", "wb"))
