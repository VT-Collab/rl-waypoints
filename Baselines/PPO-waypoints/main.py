import os
import glob
import time
import datetime
import itertools

import torch
import numpy as np

import gym

from PPO import PPO
from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.controllers import load_controller_config

from scipy.spatial.transform import Rotation
import pickle
import os, sys


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    run_name = 'runs/ppo_' + datetime.datetime.now().strftime("%H-%M")
    writer = SummaryWriter(run_name)

    has_continuous_action_space = True  # continuous action space; else discrete

    # parameters you may need to tune for each environment
    # probably you just need to tune the num_waypoints
    num_waypoints = 3                   # number of waypoints
    action_std = 0.1                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.05                # minimum action_std (stop decay after action_std <= min_action_std)

    # parameters you will not need to tune
    log_freq = num_waypoints * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    action_std_decay_freq = int(2e5)  # action_std decay frequency (in num timesteps)

    #####################################################

    ################ PPO hyperparameters ################

    # default values recommended by repository
    # probably fine to leave these alone
    update_timestep = num_waypoints * 10      # update policy every n timesteps
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
        env_name="Door",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True, # toggle this when we want to render
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        use_latch=False,
    )

    # state space dimension
    # modify this based on the state you are using for the current environment
    state_dim = 6
    # action space dimension
    # modify this if you want to rotate the gripper
    action_dim = 4

    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("number of waypoints per episode : ", num_waypoints)
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
    updates = 0
    for i_episode in itertools.count(1):

        episode_reward = 0
        obs = env.reset()

        # store home position
        robot_home = np.copy(obs['robot0_eef_pos'])

        for _ in range(num_waypoints):

            # identify the state the policy should condition on
            # this will likely change for each different environment
            state = list(obs['robot0_eef_pos']) + list(obs['door_pos'])
            state = np.array(state)  
            segment_reward = 0

            # get segment waypoint
            waypoint = ppo_agent.select_action(state)

            # center waypoint at the home position
            waypoint_normalized = np.copy(waypoint)
            waypoint_normalized[0:3] += robot_home

            # number of steps per waypoint
            for timestep in range(40):

                env.render()    # toggle this when we don't want to render

                # get error between waypoint and current position and orientation around z axis
                error = waypoint_normalized[0:3] - obs['robot0_eef_pos']
                error_angle = 0 - Rotation.from_quat(obs['robot0_eef_quat']).as_euler('xyz')[-1]

                # # if you ever need the rotation around the z axis of an object or 
                # # of the robot end-effector, use this to get the angle:
                # angle = Rotation.from_quat(obs['robot0_eef_quat']).as_euler('xyz')[-1]

                if timestep > 25:
                    # open or close the gripper
                    full_action = np.array([0.]*6 + [waypoint_normalized[3]])
                else:
                    # move to the waypoint
                    full_action = np.array(list(10. * error) + [0., 0., 1.0 * error_angle, 0.])

                # take action
                obs, reward, _, _ = env.step(full_action)
                total_numsteps += 1
                segment_reward += reward
                episode_reward += reward

            # saving reward and is_terminals
            # I scaled up the reward here. It is nonstandard, but I think the differences were too small for the 
            # robot to pick up on them. This led to better performance.
            ppo_agent.buffer.rewards.append(segment_reward * 100)
            ppo_agent.buffer.is_terminals.append(False)
            updates += 1

            # update PPO agent
            if updates % update_timestep == 0:
                print("here")
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and updates % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        writer.add_scalar('reward', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, reward: {}".format(i_episode, total_numsteps, round(episode_reward, 2)))


    # print total training time
    print("============================================================================================")
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    