import numpy as np
import torch
from memory import MyMemory
from method import Method
import datetime
import robosuite as suite
from robosuite.controllers import load_controller_config
from torch.utils.tensorboard import SummaryWriter
import pickle
from scipy.spatial.transform import Rotation


# more clever activation functions?
# overshooting goal to more quickly get out of lower region
# but the next waypoint should deal with that?


# env name
env_name = "Stack"

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name=env_name,
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
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
total_waypoints = 3
batch_size = 30
n_batches = 100
n_init = 50
n_per_waypoint = 400


# Logger
run_name = 'runs/ours_' + env_name + '_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)


# Main loop
updates = 0
total_interactions = 0
my_agents = []
my_waypoints = []

for n_waypoint in range(total_waypoints):

    # Agent
    agent = Method(state_dim, obs_dim)

    # Memory
    memory = MyMemory()

    # Best Waypoint
    best_waypoint = None
    best_reward = -np.inf

    for n_interaction in range(1, n_per_waypoint+1):

        # reset to home position
        total_interactions += 1
        obs = env.reset()
        cubeA_pos = obs['cubeA_pos']
        cubeA_angle = Rotation.from_quat(obs['cubeA_quat']).as_euler('xyz')[-1]
        cubeB_pos = obs['cubeB_pos']
        cubeB_angle = Rotation.from_quat(obs['cubeB_quat']).as_euler('xyz')[-1]
        cube_state = np.array(list(cubeA_pos) + [cubeA_angle] 
                            + list(cubeB_pos) + [cubeB_angle])

        # occasionally reset a reward model
        if n_interaction < (n_per_waypoint-50) and np.random.rand() < 0.05:
            agent.reset_model(np.random.randint(agent.n_models))

        # train reward models
        if len(memory) > batch_size:
            for _ in range(n_batches):
                critic_loss = agent.update_parameters(memory, batch_size)
                writer.add_scalar('model/critic', critic_loss, updates)
                updates += 1

        ## interaction trajectory
        # previous trajectory
        xi_full = []
        for widx, prev_agent in enumerate(my_agents):
            init_xi = my_waypoints[widx]
            prev_xi = prev_agent.traj_opt(init_xi, cube_state, prev_agent.n_models)
            xi_full.append(prev_xi)
        # new waypoint
        xi = agent.sample_waypoint()
        if n_interaction > max([n_init, batch_size]):
            xi = agent.traj_opt(best_waypoint, cube_state)
            xi[-1] = np.random.choice([-1.0, 1.0])
        xi_full.append(xi)
        traj = np.concatenate((xi, cube_state), -1)

        # execute trajectory to get reward
        episode_reward = 0
        for waypoint in xi_full:
            waypoint_reward = 0
            for timestep in range(50):

                # env.render()    # toggle this when we don't want to render

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
                waypoint_reward += reward

        # update best waypoint
        if waypoint_reward > best_reward:
            best_reward = waypoint_reward
            best_waypoint = np.copy(xi)

        # record trajectory reward pair
        memory.push(traj, waypoint_reward / 10.)
        writer.add_scalar('reward', episode_reward, total_interactions)
        print("Episode: {}, Reward: {}, WP-Reward: {}, Pred-Reward: {}".
            format(total_interactions, round(episode_reward, 2), 
                        round(waypoint_reward, 2), round(10 * agent.get_avg_reward(traj), 2)))

    # save the trained agent for the previous waypoint
    my_waypoints.append(best_waypoint)
    pickle.dump(my_waypoints, open(run_name + "/init_traj.pkl", "wb"))
    agent.save_models(run_name + "/reward_model_" + str(n_waypoint))
    my_agents.append(agent)
