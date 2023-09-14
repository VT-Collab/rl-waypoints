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
state_dim = 3
n_waypoints = 4
batch_size = 30
init_trajs = 40
n_samples = 1


# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Door",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    reward_shaping=True,
    control_freq=10,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    initialization_noise=None,
    use_latch=False,
)

# Agent
agent = Method(state_dim, n_waypoints)
agent.set_n_samples(n_samples)

# Memory
memory = MyMemory()

# Logger
run_name = 'runs/ours_door_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)
reward_data = []


# Main loop
total_steps = 0
for i_episode in range(1, 501):

    # initialize variables
    episode_reward = 0
    segment_reward = np.zeros((n_waypoints,1))
    obs = env.reset()
    robot_home = np.copy(obs['robot0_eef_pos'])
    
    # select optimal trajectory
    traj = 0.5*(np.random.rand(n_waypoints*state_dim)-0.5)
    if i_episode > init_trajs:
        traj = agent.traj_opt()
    traj = np.reshape(traj, (n_waypoints,state_dim))
    traj_mat = traj + robot_home
    
    for widx in range(n_waypoints):

        for timestep in range(25):

            # env.render()    # toggle this when we don't want to render

            if len(memory) > batch_size:
                for _ in range(1):
                    critic_loss = agent.update_parameters(memory, batch_size)
                    writer.add_scalar('model/critic', critic_loss, total_steps)

            # convert traj to actions
            state = obs['robot0_eef_pos']
            error = traj_mat[widx, :3] - state
            full_action = np.array(list(10. * error) + [0.]*4)
            
            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward
            total_steps += 1

        segment_reward[widx] = episode_reward / 10.0

    memory.push(traj, segment_reward)
    if episode_reward > agent.best_reward:
        agent.set_init(traj, episode_reward)

    writer.add_scalar('reward', episode_reward, i_episode)
    print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
    reward_data.append(episode_reward)
    pickle.dump(reward_data, open(run_name + "/rewards.pkl", "wb"))
