import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: False)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=120, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Stack",
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
env_action_space = gym.spaces.Box(
            low=-0.5,
            high=+0.5,
            shape=(4,),
            dtype=np.float64)
agent = SAC(9, env_action_space, args)

#Tensorboard
run_name = 'runs/sac_stack_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)
reward_data = []

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in range(1, 1001):
    episode_reward = 0
    episode_steps = 0
    obs = env.reset()
    robot_home = np.zeros((4,))
    robot_home[:3] = np.copy(obs['robot0_eef_pos'])
    waypoint = None

    # number of waypoints
    for _ in range(4):

        # initialize segment
        state = list(obs['robot0_eef_pos']) + list(obs['cubeA_pos']) + list(obs['cubeB_pos'])
        state = np.array(state)
        start_state = np.copy(state)
        segment_reward = 0

        # get segment waypoint
        if i_episode < 40:
            waypoint = 0.5*(np.random.rand(4)-0.5)
            waypoint[3] *= 2
        else:
            waypoint = agent.select_action(state)
        waypoint_normalized = waypoint + robot_home
        waypoint_normalized[3] *= 2

        # number of steps per waypoint
        for timestep in range(40):

            # env.render()    # toggle this when we don't want to render

            # compute action for low-level controller
            state = list(obs['robot0_eef_pos']) + list(obs['cubeA_pos']) + list(obs['cubeB_pos'])
            state = np.array(state)
            error = waypoint_normalized[:3] - state[:3]
            if timestep < 15:
                # give some time to open / close the gripper
                full_action = np.array(list(0.0 * error) + [0.]*3 + [waypoint_normalized[3]])
            else:
                # normal actions
                full_action = np.array(list(10. * error) + [0.]*3 + [waypoint_normalized[3]])

            # train sac agent
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            # take step and update
            obs, reward, done, _ = env.step(full_action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            segment_reward += reward

        # get the final state of the segment (ideally at the waypoint)
        next_state = list(obs['robot0_eef_pos']) + list(obs['cubeA_pos']) + list(obs['cubeB_pos'])
        next_state = np.array(next_state)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1.0 if episode_steps == 100 else float(not done)

        # push to the memory buffer
        memory.push(start_state, waypoint, 100*segment_reward, next_state, mask)

    writer.add_scalar('reward', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    reward_data.append(episode_reward)
    pickle.dump(reward_data, open(run_name + "/rewards.pkl", "wb"))
