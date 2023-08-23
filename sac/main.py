import numpy as np
import torch
from memory import MyMemory
from sac import SAC
# import gym_example
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import robosuite as suite
from robosuite.controllers import load_controller_config


# training parameters
batch_size = 256

# # Environment
# env = gym.make('reachdense-v0')

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Door", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    controller_configs=controller_config,
    has_renderer=True,
    reward_shaping=True,
    control_freq=10,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    initialization_noise=None,
)

# Agent
agent = Method(traj_dim=12, state_dim=3)

# Memory
memory = MyMemory()

# Logger
run_name = 'runs/ours_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)


# Main loop
total_steps = 0
for i_episode in range(1, 501):

    # initialize variables
    # timestep = 0
    episode_reward = 0
    done, truncated = False, False
    obs = env.reset()
    # xi = np.zeros((30,2))

    # select optimal trajectory
    traj = 1.0*(np.random.rand(12)-0.5)
    if i_episode > 40:
        traj = agent.traj_opt()
    traj_mat = np.reshape(traj, (4,3)) + obs['robot0_eef_pos']
  
    # while not done and not truncated:
    for t in range(200):

        # train the models
        if len(memory) > batch_size:
            for _ in range(1):
                critic_loss = agent.update_parameters(memory, batch_size)
                writer.add_scalar('model/critic', critic_loss, total_steps)

        widx = int(np.floor(t / 50))
        error = traj_mat[widx, :] - obs['robot0_eef_pos']
        action = list(error) + [0.]*4
        action = 10*np.array(action)
        # first 3 = xyz, last one is open close
        obs, reward, done, info = env.step(action)
        if reward >= 0.5:
            print("success! It's open")

        # # visualizer
        # xi[timestep,:] = np.copy(state)
        env.render()  # render on display

        # take action and record results
        # next_state, reward, done, truncated, _ = env.step(action)
        # state = next_state
        episode_reward += reward
        # timestep += 1
        total_steps += 1

    memory.push(traj, episode_reward)
    if episode_reward > agent.best_reward:
        agent.set_init(traj, episode_reward)
        print(episode_reward, "here")
    writer.add_scalar('reward', episode_reward, i_episode)
    print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
    
    # if i_episode % 1 == 0:
    #     plt.plot(xi[:,0], xi[:,1], 'bo-')
    #     plt.plot(env.goal1[0], env.goal1[1], 'gs')
    #     plt.plot(env.goal2[0], env.goal2[1], 'gs')
    #     plt.axis([-1.5, 1.5, -1.5, 1.5])
    #     plt.savefig(run_name + "/" + str(i_episode) + ".png")
    #     plt.clf()



# training parameters
batch_size = 256

# Environment
env = gym.make('reachdense-v0')

# Agent
agent = SAC(state_dim=2, action_dim=2)

# Memory
memory = MyMemory()

# Logger
run_name = 'runs/sac_' + datetime.datetime.now().strftime("%H-%M")
writer = SummaryWriter(run_name)


# Main loop
total_steps = 0
for i_episode in range(1, 501):

    timestep = 0
    episode_reward = 0
    done, truncated = False, False
    state, _ = env.reset()
    xi = np.zeros((30,2))

    while not done and not truncated:

        # train the models
        if len(memory) > batch_size:
            critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, batch_size)
            writer.add_scalar('model/critic1', critic_1_loss, total_steps)
            writer.add_scalar('model/critic2', critic_2_loss, total_steps)
            writer.add_scalar('model/actor', policy_loss, total_steps)

        # use policy to select action
        # choose random action for a while
        if i_episode < 40: 
            action = env.action_space.sample()
        # then use the learned policy
        else:
            action = agent.select_action(state)

        # visualizer
        xi[timestep,:] = np.copy(state)

        # transition to next_state and store data
        next_state, reward, done, truncated, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        timestep += 1
        total_steps += 1
        
    writer.add_scalar('reward', episode_reward, i_episode)
    print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))
    
    if i_episode % 1 == 0:
        plt.plot(xi[:,0], xi[:,1], 'bo-')
        plt.plot(env.goal1[0], env.goal1[1], 'gs')
        plt.plot(env.goal2[0], env.goal2[1], 'gs')
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.savefig(run_name + "/" + str(i_episode) + ".png")
        plt.clf()