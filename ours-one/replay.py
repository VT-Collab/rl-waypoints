import numpy as np
import torch
from method import Method
import robosuite as suite
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation
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
)
obs = env.reset()
robot_home = np.copy(obs['robot0_eef_pos'])


state_dim = 5
obs_dim = 8
total_waypoints = 2


# Load Name
run_name = 'runs/ours_' + env_name + '_' + '22-58'


# Load trained agents
my_agents = []
for n_waypoint in range(total_waypoints):
    agent = Method(state_dim, obs_dim)
    agent.load_models(run_name + "/reward_model_" + str(n_waypoint))
    my_agents.append(agent)
my_waypoints = pickle.load(open(run_name + "/init_traj.pkl", "rb"))


# Main loop
for n_interaction in range(1, 51):

    # reset to home position
    obs = env.reset()
    cubeA_pos = obs['cubeA_pos']
    cubeA_angle = Rotation.from_quat(obs['cubeA_quat']).as_euler('xyz')[-1]
    cubeB_pos = obs['cubeB_pos']
    cubeB_angle = Rotation.from_quat(obs['cubeB_quat']).as_euler('xyz')[-1]
    cube_state = np.array(list(cubeA_pos) + [cubeA_angle] 
                        + list(cubeB_pos) + [cubeB_angle])

    ## interaction trajectory
    xi_full = []
    for widx, prev_agent in enumerate(my_agents):
        init_xi = my_waypoints[widx]
        prev_xi = prev_agent.traj_opt(init_xi, cube_state, prev_agent.n_models)
        xi_full.append(prev_xi)

    # execute trajectory to get reward
    episode_reward = 0
    for waypoint in xi_full:
        for timestep in range(40):

            env.render()    # toggle this when we don't want to render

            # convert traj to actions
            state = obs['robot0_eef_pos'] - robot_home
            angle = Rotation.from_quat(obs['robot0_eef_quat']).as_euler('xyz')[-1]
            error = waypoint[:3] - state
            error_angle = waypoint[3] - angle
            if timestep > 25:
                # give some time to open / close the gripper
                full_action = np.array([0.]*6 + [waypoint[-1]])
            else:
                # normal actions
                full_action = np.array(list(10. * error) + [0, 0, 10. * error_angle, 0])

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward

    # print trajectory reward pair
    print("Episode: {}, Reward: {}".format(n_interaction, round(episode_reward, 2)))
