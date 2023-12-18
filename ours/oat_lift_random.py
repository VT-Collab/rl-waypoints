# # import numpy as np
# # import torch
# # from memory import MyMemory
# # from oat_method_random import OAT
# # import gym
# # import datetime
# # from torch.utils.tensorboard import SummaryWriter
# # import robosuite as suite
# # from robosuite.controllers import load_controller_config
# # import pickle
# # import argparse

# # def run_ours(args):
# #     # training parameters
# #     save_data = {'episode': [], 'reward': [], 'best_traj': []}
# #     if args.object == 'test':
# #         save_name = 'Lift/' + args.run_num 
# #     batch_size = 30

# #     # load default controller parameters for Operational Space Control (OSC)
# #     controller_config = load_controller_config(default_controller="OSC_POSE")

# #     # create environment instance
# #     env = suite.make(
# #         env_name="Lift", # try with other tasks like "Stack" and "Door"
# #         robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
# #         controller_configs=controller_config,
# #         has_renderer=True,
# #         reward_shaping=True,
# #         control_freq=10,
# #         has_offscreen_renderer=False,
# #         use_camera_obs=False,
# #         initialization_noise=None,
# #         # use_latch=False,
# #     )

# #     # Agent
# #     num_wp = args.num_wp
# #     wp_id = 1
# #     obs = env.reset()
# #     objs = obs['cube_pos']
# #     agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name)


# #     # Memory
# #     memory = MyMemory()

# #     # Logger
# #     run_name = 'runs/ours_' + args.run_num + datetime.datetime.now().strftime("%H-%M")
# #     writer = SummaryWriter(run_name)
# #     epoch_wp = 200
# #     EPOCHS = epoch_wp*num_wp


# #     # Main loop
# #     total_steps = 0
# #     mean_episode_reward = 0
# #     std_episode_reward = 0
# #     best_wps = []
# #     best_wp = []    
# #     for i_episode in range(1, EPOCHS):
# #         if i_episode % epoch_wp == 0:
# #             # best_wps = agent.best_traj
# #             # best_wp = np.array(best_wps).squeeze()
# #             # agent.eval_best_model(memory, save_name)
# #             agent.save_model(save_name)

# #             wp_id += 1
# #             agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name)
# #              # Memory
# #             memory = MyMemory()

# #         if np.random.rand()<0.05 and i_episode<150:
# #             agent.reset_model(np.random.randint(10))

# #         i_episode = i_episode%epoch_wp


# #         # initialize variables
# #         episode_reward = 0
# #         done, truncated = False, False
# #         obs = env.reset()
# #         objs = obs['cube_pos']

# #         # select optimal trajectory
# #         traj_full = agent.traj_opt(i_episode, objs)
        
# #         traj = traj_full[-4:]
        
# #         traj_mat = np.reshape(traj_full, (wp_id,4))[:, :3] + obs['robot0_eef_pos']
# #         gripper_mat = np.reshape(traj_full, (wp_id, 4))[:, 3:] 

# #         if len(memory) > batch_size:
# #                 for _ in range(100):
# #                     critic_loss = agent.update_parameters(memory, batch_size)
# #                     writer.add_scalar('model/critic', critic_loss, total_steps)
        
# #         time_s = 0
# #         for timestep in range(wp_id*50):

# #             # if i_episode>350:
# #             env.render()    # toggle this when we don't want to render

# #             # convert traj to actions
# #             state = obs['robot0_eef_pos']
# #             g_state = obs['robot0_gripper_qpos']
# #             widx = int(np.floor(timestep / (50)))
# #             error = traj_mat[widx, :] - state
# #             # error_g = gripper_mat[widx, :] - g_state
# #             if time_s >= 40:
# #                 full_action = np.array([0.]*6 + list(gripper_mat[widx]))
# #             else:
# #                 full_action = np.array(list(10. * error) + [0.]*4)
# #             time_s += 1
# #             if time_s >= 50:
# #                 time_s = 1

# #             # take step
# #             obs, reward, done, _ = env.step(full_action)
# #             episode_reward += reward
# #             total_steps += 1

# #         memory.push(np.concatenate((traj, objs)), episode_reward)
# #         save_data['episode'].append(i_episode)
# #         save_data['reward'].append(episode_reward)

# #         if episode_reward > agent.best_reward:
# #             agent.set_init(traj_full, episode_reward)
# #             print(episode_reward, "new best trajectory")
# #             print(traj)
# #             agent.save_model(save_name)
# #             save_data['best_traj'].append(episode_reward)
# #             pickle.dump(traj_full, open('models/' + save_name + '/traj.pkl', 'wb'))

# #         writer.add_scalar('reward', episode_reward, i_episode)
# #         print("wp_id: {}, Episode: {}, Reward: {}, Predicted Reward: {}".format(wp_id, i_episode, round(episode_reward, 2), agent.get_avg_reward(traj)))

# #     pickle.dump(save_data, open('models/' + save_name + '/data.pkl', 'wb'))



# # if __name__=='__main__':
# #     p = argparse.ArgumentParser()
# #     p.add_argument('--run_num', type=str, default='test')
# #     p.add_argument('--object', type=str, default='test')
# #     p.add_argument('--n_inits', type=int, default=1)
# #     p.add_argument('--num_wp', type=int, default=5)
# #     args = p.parse_args()
# #     run_ours(args)


# import numpy as np
# import torch
# from memory import MyMemory
# from oat_method_random import OAT
# import gym
# import datetime
# from torch.utils.tensorboard import SummaryWriter
# import robosuite as suite
# from robosuite.controllers import load_controller_config
# import pickle
# import argparse

# def run_ours(args):
#     # training parameters
#     save_data = {'episode': [], 'reward': [], 'best_traj': []}
#     if args.object == 'test':
#         save_name = 'Lift/' + args.run_num 
#     batch_size = 30

#     # load default controller parameters for Operational Space Control (OSC)
#     controller_config = load_controller_config(default_controller="OSC_POSE")

#     # create environment instance
#     env = suite.make(
#         env_name="Lift", # try with other tasks like "Stack" and "Door"
#         robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#         controller_configs=controller_config,
#         has_renderer=True,
#         reward_shaping=True,
#         control_freq=10,
#         has_offscreen_renderer=False,
#         use_camera_obs=False,
#         initialization_noise=None,
#         # use_latch=False,
#     )

#     # Agent
#     num_wp = args.num_wp
#     wp_id = 1
#     obs = env.reset()
#     objs = obs['cube_pos']
#     agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name)


#     # Memory
#     memory = MyMemory()

#     # Logger
#     run_name = 'runs/ours_' + args.run_num + datetime.datetime.now().strftime("%H-%M")
#     writer = SummaryWriter(run_name)
#     epoch_wp = 200
#     EPOCHS = epoch_wp*num_wp


#     # Main loop
#     total_steps = 0
#     mean_episode_reward = 0
#     std_episode_reward = 0
#     best_wps = []
#     best_wp = []    
#     for i_episode in range(1, EPOCHS):
#         if i_episode % epoch_wp == 0:
#             # best_wps = agent.best_traj
#             # best_wp = np.array(best_wps).squeeze()
#             # agent.eval_best_model(memory, save_name)
#             agent.save_model(save_name)

#             wp_id += 1
#             agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name)
#              # Memory
#             memory = MyMemory()

#         if np.random.rand()<0.05 and i_episode<150:
#             agent.reset_model(np.random.randint(10))

#         i_episode = i_episode%epoch_wp


#         # initialize variables
#         episode_reward = 0
#         done, truncated = False, False
#         obs = env.reset()
#         objs = obs['cube_pos']

#         # select optimal trajectory
#         traj_full = agent.traj_opt(i_episode, objs)
        
#         traj = traj_full[-4:]
        
#         traj_mat = np.reshape(traj_full, (wp_id,4))[:, :3] + obs['robot0_eef_pos']
#         gripper_mat = np.reshape(traj_full, (wp_id, 4))[:, 3:] 

#         if len(memory) > batch_size:
#                 for _ in range(100):
#                     critic_loss = agent.update_parameters(memory, batch_size)
#                     writer.add_scalar('model/critic', critic_loss, total_steps)
        
#         time_s = 0
#         for timestep in range(wp_id*50):

#             # if i_episode>350:
#             env.render()    # toggle this when we don't want to render

#             # convert traj to actions
#             state = obs['robot0_eef_pos']
#             g_state = obs['robot0_gripper_qpos']
#             widx = int(np.floor(timestep / (50)))
#             error = traj_mat[widx, :] - state
#             # error_g = gripper_mat[widx, :] - g_state
#             if time_s >= 40:
#                 full_action = np.array([0.]*6 + list(gripper_mat[widx]))
#             else:
#                 full_action = np.array(list(10. * error) + [0.]*4)
#             time_s += 1
#             if time_s >= 50:
#                 time_s = 1

#             # take step
#             obs, reward, done, _ = env.step(full_action)
#             episode_reward += reward
#             total_steps += 1

#         memory.push(np.concatenate((traj, objs)), episode_reward)
#         save_data['episode'].append(i_episode)
#         save_data['reward'].append(episode_reward)

#         if episode_reward > agent.best_reward:
#             agent.set_init(traj_full, episode_reward)
#             print(episode_reward, "new best trajectory")
#             # print(traj)
#             agent.save_model(save_name)
#             save_data['best_traj'].append(episode_reward)
#             pickle.dump(traj_full, open('models/' + save_name + '/traj.pkl', 'wb'))

#         writer.add_scalar('reward', episode_reward, i_episode)
#         print("wp_id: {}, Episode: {}, Reward: {}, Predicted Reward: {}".format(wp_id, i_episode, round(episode_reward, 2), agent.get_avg_reward(traj)))

#     pickle.dump(save_data, open('models/' + save_name + '/data.pkl', 'wb'))



# if __name__=='__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('--run_num', type=str, default='test')
#     p.add_argument('--object', type=str, default='test')
#     p.add_argument('--n_inits', type=int, default=1)
#     p.add_argument('--num_wp', type=int, default=5)
#     args = p.parse_args()
#     run_ours(args)


import numpy as np
import torch
from memory import MyMemory
from oat_method_random import OAT
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
import argparse

def run_ours(args):
    # training parameters
    save_data = {'episode': [], 'reward': [], 'best_traj': []}
    if args.object == 'test':
        save_name = 'Stack/' + args.run_num 
    batch_size = 30

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # create environment instance
    env = suite.make(
        env_name="Stack", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=controller_config,
        has_renderer=args.render,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        initialization_noise=None,
        # use_latch=False,
    )

    # Agent
    num_wp = args.num_wp
    wp_id = 1
    obs = env.reset()
    # objs = obs['cube_pos']
    objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
    agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name)


    # Memory
    memory = MyMemory()

    # Logger
    run_name = 'runs/ours_' + args.run_num + datetime.datetime.now().strftime("%H-%M")
    writer = SummaryWriter(run_name)
    epoch_wp = 300
    EPOCHS = epoch_wp*num_wp


    # Main loop
    total_steps = 0
    mean_episode_reward = 0
    std_episode_reward = 0
    best_wps = []
    best_wp = []    
    for i_episode in range(1, EPOCHS):
        if i_episode % epoch_wp == 0:
            agent.save_model(save_name)

            wp_id += 1
            agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name)
             # Memory
            memory = MyMemory()


        i_episode = i_episode%epoch_wp

        if np.random.rand()<0.05 and i_episode<250 and i_episode>1:
            agent.reset_model(np.random.randint(10))

        # initialize variables
        episode_reward = 0
        done, truncated = False, False
        obs = env.reset()
        objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1)

        # select optimal trajectory
        traj_full = agent.traj_opt(i_episode, objs)
        
        traj = traj_full[-4:]
        
        traj_mat = np.reshape(traj_full, (wp_id,4))[:, :3] + obs['robot0_eef_pos']
        gripper_mat = np.reshape(traj_full, (wp_id, 4))[:, 3:] 

        if len(memory) > batch_size:
                for _ in range(100):
                    critic_loss = agent.update_parameters(memory, batch_size)
                    writer.add_scalar('model/critic', critic_loss, total_steps)
        
        time_s = 0
        train_reward = 0
        for timestep in range(wp_id*50):

            # if i_episode>350:
            if args.render:
                env.render()    # toggle this when we don't want to render

            # convert traj to actions
            state = obs['robot0_eef_pos']
            g_state = obs['robot0_gripper_qpos']
            widx = int(np.floor(timestep / (50)))
            error = traj_mat[widx, :] - state
            # error_g = gripper_mat[widx, :] - g_state
            if time_s >= 40:
                full_action = np.array([0.]*6 + list(gripper_mat[widx]))
            else:
                full_action = np.array(list(10. * error) + [0.]*4)
            time_s += 1
            if time_s >= 50:
                time_s = 1

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward
            if timestep//50 == wp_id-1:
                train_reward += reward
            total_steps += 1

        memory.push(np.concatenate((traj, objs)), train_reward)
        save_data['episode'].append(i_episode)
        save_data['reward'].append(episode_reward)

        
        if train_reward > agent.best_reward:
            agent.set_init(traj_full, train_reward)
            agent.save_model(save_name)
            save_data['best_traj'].append(episode_reward)
            pickle.dump(traj_full, open('models/' + save_name + '/traj.pkl', 'wb'))

        if i_episode > 10:
            mean_episode_reward = np.mean(save_data['reward'][-10:])
            std_episode_reward = np.std(save_data['reward'][-10:])
        
        writer.add_scalar('reward', episode_reward, i_episode)
        print("wp_id: {}, Episode: {}, Reward_full: {}; Reward: {}, Predicted: {}".format(wp_id, i_episode, round(episode_reward, 2), round(train_reward, 2), round(agent.get_avg_reward(traj), 2)))

    pickle.dump(save_data, open('models/' + save_name + '/data.pkl', 'wb'))



if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')
    p.add_argument('--n_inits', type=int, default=1)
    p.add_argument('--num_wp', type=int, default=5)
    p.add_argument('--render', action='store_true', default=False)
    args = p.parse_args()
    run_ours(args)