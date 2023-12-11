import os
import glob
import time
import datetime

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
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = 'door'

    run_name = 'runs/ppo_' + datetime.datetime.now().strftime("%H-%M")
    writer = SummaryWriter(run_name)

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # print("training environment name : " + env_name)

    # env = gym.make(env_name, continuous=True, render_mode="None")

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # create environment instance
    env = suite.make(
        env_name="Door", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=controller_config,
        has_renderer=False,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        use_latch=True,
    )
    env_action_space = gym.spaces.Box(
                low=-0.3,
                high=+0.3,
                shape=(6,),
                dtype=np.float64)

    # state space dimension
    state_dim = 6

    # action space dimension
    if has_continuous_action_space:
        action_dim = env_action_space.shape[0]
    else:
        action_dim = env_action_space.n

    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
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

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    best_reward = -np.inf
    while time_step <= max_training_timesteps:

        obs = env.reset()
        current_ep_reward = 0

        traj = []

        for t in range(1, max_ep_len+1):

            lin = obs['robot0_eef_pos']
            quat = obs['robot0_eef_quat']
            r = R.from_quat(quat)
            ang_eul = r.as_euler('xyz', degrees=False)
            pos = np.concatenate((lin, ang_eul))


            # env.render()    # toggle this when we don't want to render

            # select action with policy
            state = pos
            action = ppo_agent.select_action(state)
            # full_action = np.array(list(action) + [0.]*4)
            full_action = np.array(list(action) + [-1])
            obs, reward, done, _ = env.step(full_action)

            traj.append(state)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # break; if the episode is over
            if done:
                break

        if current_ep_reward > best_reward:
            if not os.path.exists('models/Door_ang/'):
                os.makedirs('models/Door_ang/')
    
            print("Trajectory Saved !!! ")
            best_reward = current_ep_reward

            pickle.dump(traj, open('models/Door_ang/traj_{}.pkl'.format(best_reward), 'wb'))

        writer.add_scalar('reward', current_ep_reward, i_episode)
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1


    # print total training time
    print("============================================================================================")
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    