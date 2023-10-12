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


# option 1: reward for one point, then optimize as normal
# option 2: bayesian optimization


## put open / close at end

# training parameters
state_dim = 4
n_waypoints = 3
batch_size = 30
init_trajs = 30
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


# Main loop
learned_waypoints = []

for widx in range(n_waypoints):
    
    # Agent
    agent = Method(state_dim, 1)
    agent.set_n_samples(n_samples)

    # Memory
    memory = MyMemory()
    updates = 0
    
    for i_episode in range(1, 101):

        # occasionally reset a reward model
        if np.random.rand() < 0.05:
            agent.reset_model(np.random.randint(agent.n_models))

        # initialize variables
        episode_reward = 0
        obs = env.reset()
        robot_home = np.zeros((4,))
        robot_home[:3] = np.copy(obs['robot0_eef_pos'])

        # select optimal trajectory
        traj = 0.5*(np.random.rand(state_dim)-0.5)
        traj[-1] *= 2.0
        if i_episode > init_trajs:
            traj = agent.traj_opt()
        traj[-1] *= 2.0
        proposed_waypoint = traj + robot_home

        # get overall trajectory
        traj_mat = copy.deepcopy(learned_waypoints)
        traj_mat.append(proposed_waypoint)

        # save robot and object trajectory
        xi_robot = []
        xi_cube = []

        for waypoint in traj_mat:

            for timestep in range(40):

                env.render()    # toggle this when we don't want to render

                # train reward models
                if len(memory) > batch_size:
                    for _ in range(1):
                        critic_loss = agent.update_parameters(memory, batch_size)
                        writer.add_scalar('model/critic', critic_loss, updates)
                        updates += 1

                # convert traj to actions
                state = obs['robot0_eef_pos']
                error = waypoint[:3] - state
                if timestep < 15:
                    # give some time to open / close the gripper
                    full_action = np.array(list(0.0 * error) + [0.]*3 + [waypoint[3]])
                else:
                    # normal actions
                    full_action = np.array(list(10. * error) + [0.]*3 + [waypoint[3]])

                # take step
                obs, reward, done, _ = env.step(full_action)
                episode_reward += reward

                # save robot and object trajectory
                xi_robot.append(list(obs['robot0_eef_pos']))
                xi_cube.append(list(obs['cube_pos']))

        memory.push(traj, episode_reward / 10.)
        if episode_reward > agent.best_reward:
            agent.set_init(traj, proposed_waypoint, episode_reward)
            pickle.dump([xi_robot, xi_cube], open(run_name + "/traj.pkl", "wb"))    

        writer.add_scalar('reward', episode_reward, i_episode)
        print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
        print(round(obs['cube_pos'][-1], 3))
        reward_data.append(episode_reward)
        pickle.dump(reward_data, open(run_name + "/rewards.pkl", "wb"))

    learned_waypoints.append(agent.best_waypoint)
