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
state_dim = 4
n_waypoints = 3
batch_size = 30
init_trajs = 40
n_samples = 1

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Lift",
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
agent = Method(state_dim, n_waypoints)
agent.set_n_samples(n_samples)

# Memory
memory = MyMemory()

# Logger
run_name = 'runs/ours_lift_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)
reward_data = []


# Main loop
updates = 0
for i_episode in range(1, 501):

    # occasionally reset a reward model
    if np.random.rand() < 0.05:
        agent.reset_model(np.random.randint(agent.n_models))

    # initialize variables
    episode_reward = 0
    obs = env.reset()
    robot_home = np.zeros((4,))
    robot_home[:3] = np.copy(obs['robot0_eef_pos'])

    # select optimal trajectory
    traj = 0.5*(np.random.rand(n_waypoints,state_dim)-0.5)
    traj[:,-1] *= 2.0
    traj = np.reshape(traj, (-1,))
    if i_episode > init_trajs:
        traj = agent.traj_opt()
    traj_mat = np.reshape(traj, (n_waypoints,state_dim)) + robot_home
    traj_mat[:,-1] *= 2.0

    # save robot and object trajectory
    xi_robot = []
    xi_cube = []

    for widx in range(n_waypoints):

        for timestep in range(40):

            # env.render()    # toggle this when we don't want to render

            # train reward models
            if len(memory) > batch_size:
                for _ in range(1):
                    critic_loss = agent.update_parameters(memory, batch_size)
                    writer.add_scalar('model/critic', critic_loss, updates)
                    updates += 1

            # convert traj to actions
            state = obs['robot0_eef_pos']
            error = traj_mat[widx, :3] - state
            if timestep < 15:
                # give some time to open / close the gripper
                full_action = np.array(list(0.0 * error) + [0.]*3 + [traj_mat[widx, 3]])
            else:
                # normal actions
                full_action = np.array(list(10. * error) + [0.]*3 + [traj_mat[widx, 3]])

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward

            # save robot and object trajectory
            xi_robot.append(list(obs['robot0_eef_pos']))
            xi_cube.append(list(obs['cube_pos']))

    memory.push(traj, episode_reward / 10.)
    if episode_reward > agent.best_reward:
        agent.set_init(traj, episode_reward)
        pickle.dump([xi_robot, xi_cube], open(run_name + "/traj.pkl", "wb"))    

    writer.add_scalar('reward', episode_reward, i_episode)
    print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
    reward_data.append(episode_reward)
    pickle.dump(reward_data, open(run_name + "/rewards.pkl", "wb"))
