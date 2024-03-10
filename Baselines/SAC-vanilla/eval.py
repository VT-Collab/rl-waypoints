import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import pickle
import os, sys
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import robosuite as suite
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# tune this to change the amount of noise in the actions
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.1)')
# tune this to change the number of timesteps per interaction
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='number of steps per episode (default: 100)')
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--run_num', type=str, default='test')
parser.add_argument('--object', type=str, default='test')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--num_eval', type=int, default=100)

# tune this to change how many random actions at the start of the RL loop
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions')
# probably don't tune any of the rest
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
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


save_data = {'episode': [], 'reward': []}
if args.object == 'test':
    save_name = 'models/' + args.env + '/' + args.run_num
else:
    save_name = 'models/' + args.env + '/' + args.object + '/' + args.run_num

if not os.path.exists(save_name):
    os.makedirs(save_name)
# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name=args.env,
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=args.render, # toggle this when we want to render
    reward_shaping=True,
    control_freq=10,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    initialization_noise=None,
    single_object_mode=2,
    object_type=args.object,
    use_latch=False,
)

obs = env.reset()
if args.env == 'Stack':
    objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
elif args.env == 'Lift':
    objs = obs['cube_pos']
elif args.env == 'PickPlace':
    objs = obs[args.object+'_pos']
elif args.env == 'NutAssembly':
    nut = 'RoundNut'# if obs['nut_id'] == 0 else 'RoundNut'
    objs = obs[nut + '_pos']

# Agent
# add dimensions to env_action_space if you also want to rotate the end-effector
env_action_space = gym.spaces.Box(
            low=-1.0,
            high=+1.0,
            shape=(4,),
            dtype=np.float64)
# first argument is the dimension of the state space, will change for each environment
agent = SAC(len(obs['robot0_eef_pos']) + len(objs), env_action_space, args)
agent.load_checkpoint(save_name + '/models', evaluate=True)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    obs = env.reset()

    if args.env == 'Stack':
            objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
    elif args.env == 'Lift':
        objs = obs['cube_pos']
    elif args.env == 'PickPlace':
        objs = obs[args.object+'_pos']
    elif args.env == 'NutAssembly':
        nut = 'RoundNut'# if obs['nut_id'] == 0 else 'RoundNut'
        objs = obs[nut + '_pos']

    # identify the state the policy should condition on
    # this will likely change for each different environment
    state = list(obs['robot0_eef_pos']) + list(objs)
    state = np.array(state)    
    
    for timestep in range(1, args.num_steps+1):

        if args.render:
            env.render()    # toggle this when we don't want to render

        
        action = agent.select_action(state)  # Sample action from policy

        # compute action for low-level controller
        full_action = list(action[0:3]) + [0.]*3 + [action[3]]
        full_action = np.array(full_action)

        # take action
        obs, reward, _, _ = env.step(full_action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        if args.env == 'Stack':
            objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
        elif args.env == 'Lift':
            objs = obs['cube_pos']
        elif args.env == 'PickPlace':
            objs = obs[args.object+'_pos']
        elif args.env == 'NutAssembly':
            nut = 'RoundNut'
            objs = obs[nut + '_pos']

        # identify the state the policy should condition on
        # this will likely change for each different environment
        next_state = list(obs['robot0_eef_pos']) + list(objs)
        next_state = np.array(next_state)

        state = next_state

    save_data['episode'].append(i_episode)
    save_data['reward'].append(episode_reward)
    pickle.dump(save_data, open(save_name + '/eval_reward.pkl', 'wb'))
    print("Saved test data to {}".format(save_name + '/eval_reward.pkl'))
    if i_episode == args.num_eval:
        exit()


