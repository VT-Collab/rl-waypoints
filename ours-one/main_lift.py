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
import copy


# training parameters
state_dim = 4
max_waypoints = 2
batch_size = 30
n_batches = 40
n_samples = 5

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    reward_shaping=True,
    control_freq=10,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    initialization_noise=None,
)


# Logger
run_name = 'runs/ours_lift_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)
reward_data = []


# Agent
agent = Method(state_dim, max_waypoints)
agent.set_n_samples(n_samples)


# Memory
memory = MyMemory()


# Main loop
updates = 0
i_episode_total = 0

for n_waypoints in range(1, max_waypoints+1):

    agent.set_init(None, -np.inf)
    
    for i_episode in range(1, 101):

        # global episode counter
        i_episode_total += 1

        # occasionally reset a reward model
        if np.random.rand() < 0.05:
            agent.reset_model(np.random.randint(agent.n_models))

        # train reward models
        if len(memory) > batch_size:
            for _ in range(n_batches):
                critic_loss = agent.update_parameters(memory, batch_size)
                writer.add_scalar('model/critic', critic_loss, updates)
                updates += 1 

        # initialize variables
        episode_reward = 0
        obs = env.reset()
        robot_home = np.zeros((4,))
        robot_home[:3] = np.copy(obs['robot0_eef_pos'])

        # select optimal trajectory
        traj = 0.5*(np.random.rand(n_waypoints*state_dim)-0.5)
        if i_episode > batch_size:
            traj = agent.traj_opt()
        traj_temp = np.reshape(traj, (n_waypoints, state_dim))
        traj_mat = np.zeros((max_waypoints, state_dim))
        traj_mat[:n_waypoints, :] = traj_temp
        traj_mat[n_waypoints:, :] = traj_temp[-1,:]
        traj_for_memory = np.reshape(traj_mat, (-1,))
        traj_mat += robot_home         

        # inner loop: rollout the trajectory and record reward
        for waypoint in traj_mat:
            for timestep in range(40):

                env.render()    # toggle this when we don't want to render

                # convert traj to actions
                state = obs['robot0_eef_pos']
                error = waypoint[:3] - state
                if timestep > 25:
                    # give some time to open / close the gripper
                    full_action = np.array(list(0.0 * error) + [0.]*3 + [waypoint[-1]])
                else:
                    # normal actions
                    full_action = np.array(list(10. * error) + [0.]*4)

                # take step
                obs, reward, done, _ = env.step(full_action)
                episode_reward += reward

        memory.push(traj_for_memory, episode_reward / 10.)
        if episode_reward > agent.best_reward:
            agent.set_init(traj, episode_reward)

        writer.add_scalar('reward', episode_reward, i_episode_total)
        print("Episode: {}, Reward: {}, Cube: {}".
            format(i_episode_total, round(episode_reward, 2), round(obs['cube_pos'][-1], 3)))
        reward_data.append(episode_reward)
        pickle.dump(reward_data, open(run_name + "/rewards.pkl", "wb"))
