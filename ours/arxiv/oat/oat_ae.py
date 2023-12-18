import numpy as np
import torch
from memory import MyMemory
from oat_method_ae import OAT
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
import argparse
from tqdm import tqdm

def run_ours(args):
    # training parameters
    save_data = {'episode': [], 'reward': [], 'best_traj': []}
    if args.object == 'test':
        save_name = args.env + '/' + args.run_num 
    else:
        save_name = args.env + '/' + args.object + '/' + args.run_num 
    batch_size = 30

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

    # Agent
    num_wp = args.num_wp
    wp_id = 1
    obs = env.reset()
    # objs = obs['cube_pos']
    if args.env == 'Stack':
        objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos'], obs['robot0_eef_pos']), axis=-1) 
    elif args.env == 'Lift':
        objs = np.concatenate((obs['cube_pos'], obs['robot0_eef_pos']), axis=-1)
    elif args.env == 'PickPlace':
        objs = np.concatenate((obs[args.object+'_pos'], obs['robot0_eef_pos']), axis=-1)
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
    for i_episode in tqdm(range(1, EPOCHS)):
        if i_episode % epoch_wp == 0:
            agent.eval_best_model(memory, save_name)
            agent.save_model(save_name)

            wp_id += 1
            agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name)
             # Memory
            memory = MyMemory()


        i_episode = i_episode%epoch_wp

        # if np.random.rand()<0.05 and i_episode<250 and i_episode>1:
        #     agent.reset_model(np.random.randint(10))

        # initialize variables
        episode_reward = 0
        done, truncated = False, False
        obs = env.reset()
        if args.env == 'Stack':
            objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos'], obs['robot0_eef_pos']), axis=-1) 
        elif args.env == 'Lift':
            objs = np.concatenate((obs['cube_pos'], obs['robot0_eef_pos']), axis=-1)
        elif args.env == 'PickPlace':
            objs = np.concatenate((obs[args.object+'_pos'], obs['robot0_eef_pos']), axis=-1)
        # select optimal trajectory
        traj_full_ae, reward_ae = agent.traj_opt(i_episode, objs)

        traj_full = traj_full_ae.cpu().detach().numpy().flatten()
        
        traj = traj_full[-4:]
        
        traj_mat = np.reshape(traj_full, (wp_id,4))[:, :3] + obs['robot0_eef_pos']
        gripper_mat = np.reshape(traj_full, (wp_id, 4))[:, 3:] 

        if len(memory) > batch_size:
                for _ in range(100):
                    critic_loss = agent.update_parameters(memory, batch_size, i_episode)
                    writer.add_scalar('model/critic', critic_loss, total_steps)
        
        time_s = 0
        train_reward = 0
        for timestep in range(wp_id*100):

            if args.render:
                env.render()    # toggle this when we don't want to render

            # convert traj to actions
            state = obs['robot0_eef_pos']
            g_state = obs['robot0_gripper_qpos']
            widx = int(np.floor(timestep / (100)))
            error = traj_mat[widx, :] - state
            # error_g = gripper_mat[widx, :] - g_state
            if timestep < 10:
                full_action = np.array(list(10. * error) + [0.]*3 + [-1.])
            elif time_s >= 85:
                full_action = np.array(list(10. * error) + [0.]*3 + list(gripper_mat[widx]))
            else:
                full_action = np.array(list(10. * error) + [0.]*4)
            time_s += 1
            if time_s >= 100:
                time_s = 1

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward
            if timestep//100 == wp_id-1:
                train_reward += reward
            total_steps += 1

        memory.push(np.concatenate((objs, traj_full[:-4]), axis=-1), traj_full[-4:], [train_reward])
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
        tqdm.write("wp_id: {}, Episode: {}, Reward_full: {}; Reward: {}, Predicted: {}".format(wp_id, i_episode, round(episode_reward, 2), round(train_reward, 2), round(reward_ae.item(), 2)))

    pickle.dump(save_data, open('models/' + save_name + '/data.pkl', 'wb'))



if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True)
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')
    p.add_argument('--n_inits', type=int, default=1)
    p.add_argument('--num_wp', type=int, default=5)
    p.add_argument('--render', action='store_true', default=False)
    args = p.parse_args()
    run_ours(args)



"""
What if we take two different networks for rewards and actor all together?
i.e we have two backprops, one for reward and one for actor
"""