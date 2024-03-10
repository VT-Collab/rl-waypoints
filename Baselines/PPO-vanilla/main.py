import os
import glob
import time
import datetime
import itertools
import argparse

import torch
import numpy as np

import gym

from PPO import PPO
from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.controllers import load_controller_config

from scipy.spatial.transform import Rotation as R
import pickle
import os, sys


################################### Training ###################################
def train(args):
    save_data = {'episode': [], 'reward': []}
    if args.object == 'test':
        save_name = 'models/' + args.env + '/' + args.run_num
    else:
        save_name = 'models/' + args.env + '/' + args.object + '/' + args.run_num

    if not os.path.exists(save_name):
        os.makedirs(save_name)
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    run_name = 'runs/ppo_' + datetime.datetime.now().strftime("%H-%M")
    writer = SummaryWriter(run_name)

    has_continuous_action_space = True  # continuous action space; else discrete

    # parameters you may need to tune for each environment
    # probably you just need to tune the max_ep_len
    max_ep_len = 50*args.num_wp         # max timesteps in one episode
    action_std = 0.1                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.05                # minimum action_std (stop decay after action_std <= min_action_std)

    # parameters you will not need to tune
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    action_std_decay_freq = int(2e5)  # action_std decay frequency (in num timesteps)

    #####################################################

    ################ PPO hyperparameters ################

    # default values recommended by repository
    # probably fine to leave these alone
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # create environment instance
    env = suite.make(
        env_name=args.env, # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=controller_config,
        has_renderer=args.render,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        single_object_mode=2,
        object_type=args.object,
        use_latch=False,
    )

    # add dimensions to env_action_space if you also want to rotate the end-effector
    env_action_space = gym.spaces.Box(
                low=-1.0,
                high=+1.0,
                shape=(4,),
                dtype=np.float64)

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

    # state space dimension
    # modify this based on the state you are using for the current environment
    state_dim = len(obs['robot0_eef_pos']) + len(objs)
    # action space dimension
    # modify this if you want to rotate the gripper
    action_dim = 4

    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    # Training Loop
    total_numsteps = 0
    for i_episode in itertools.count(1):

        episode_reward = 0
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

        for timestep in range(1, max_ep_len+1):
            if args.render:
                env.render()    # toggle this when we don't want to render

            # select action with policy
            action = ppo_agent.select_action(state)

            # compute action for low-level controller
            full_action = list(action[0:3]) + [0.]*3 + [action[3]]
            full_action = np.array(full_action)

            # take action
            obs, reward, _, _ = env.step(full_action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(False)

            total_numsteps += 1
            episode_reward += reward

            # update PPO agent
            if total_numsteps % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and total_numsteps % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

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

        writer.add_scalar('reward', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, timestep, round(episode_reward, 2)))
        save_data['episode'].append(i_episode)
        save_data['reward'].append(episode_reward)
        pickle.dump(save_data, open(save_name + '/data.pkl', 'wb'))
        ppo_agent.save(save_name + '/models')

        if i_episode==args.num_episodes:
            break


    # print total training time
    print("============================================================================================")
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True)
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')
    p.add_argument('--num_wp', type=int, default=5)
    p.add_argument('--render', action='store_true', default=False)
    p.add_argument('--num_episodes', type=int, default=999)
    args = p.parse_args()
    train(args)
    