import numpy as np
import torch
from memory import MyMemory
from method import Method
import datetime
import robosuite as suite
from robosuite.controllers import load_controller_config
import datetime
from torch.utils.tensorboard import SummaryWriter
import pickle

# env name
env_name = "Stack"

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name=env_name,
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    reward_shaping=True,
    control_freq=10,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    initialization_noise=None,
    reward_scale=None,
)
obs = env.reset()
robot_home = np.copy(obs['robot0_eef_pos'])


state_dim = 4
total_waypoints = 3
batch_size = 30
n_batches = 30
n_init = 30


# Logger
run_name = 'runs/ours_' + env_name + '_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)


# Agent
agent = Method(state_dim, total_waypoints)


# Memory
memory = MyMemory()
reward_data = []


# Main loop
updates = 0
total_interactions = 0

for n_waypoint in range(total_waypoints):
    for n_interaction in range(1, 101):

        # reset to home position
        total_interactions += 1
        obs = env.reset()

        # occasionally reset a reward model
        if np.random.rand() < 0.05:
            agent.reset_model(np.random.choice(agent.n_models))

        # train reward models
        if len(memory) > batch_size:
            for _ in range(n_batches):
                critic_loss = agent.update_parameters(memory, batch_size)
                writer.add_scalar('model/critic', critic_loss, updates)
                updates += 1

        # interaction trajectory
        xi_full = np.zeros((total_waypoints, state_dim))
        xi_full[:, :] = 0.5*(np.random.rand(state_dim)-0.5)
        if n_interaction < n_init:
            widx = n_waypoint
            rand_waypoint = True
        else:
            widx = n_waypoint + 1
            rand_waypoint = False
        if widx > 0:
            xi = agent.traj_opt(widx)
            xi = np.reshape(xi, (-1, state_dim))
            xi_full[:widx, :] = xi
            if not rand_waypoint:
                xi_full[widx:, :] = xi[-1, :]
        traj = np.reshape(np.copy(xi_full), (-1, ))
        print("predicted reward: ", agent.get_avg_reward(traj))

        # execute trajectory to get reward
        episode_reward = 0
        for waypoint in xi_full:
            for timestep in range(40):

                env.render()    # toggle this when we don't want to render

                # convert traj to actions
                state = obs['robot0_eef_pos'] - robot_home
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

        # record trajectory reward pair
        memory.push(traj, episode_reward / 10.)
        writer.add_scalar('reward', episode_reward, total_interactions)
        print("Episode: {}, Reward: {}".format(total_interactions, round(episode_reward, 2)))
        reward_data.append(episode_reward)
        pickle.dump(reward_data, open(run_name + "/rewards.pkl", "wb"))