import numpy as np
import torch
from memory import MyMemory
from seed_method_cont import OAT
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
    if args.object == 'test':
        save_name = args.env + '/' + args.run_num 
    else:
        save_name = args.env + '/' + args.object + '/' + args.run_num 
    
    if not args.load_models:
        save_data = {'episode': [], 'reward': [], 'best_traj': [], 'writer_counter':None}
    else: 
        save_data = pickle.load(open('models/' + save_name + '/data.pkl', 'rb'))
        
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
    )

    # Agent
    num_wp = args.num_wp
    wp_id = args.start_wp
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



    # Memory
    memory = MyMemory()

    # Logger
    if not args.load_models:
        run_name = 'runs/ours_{}_{}'.format(args.run_num, datetime.datetime.now().strftime("%H-%M"))
        writer_counter = 0
    else:
        run_name = 'runs/ours_{}_{}'.format(args.run_num, args.run_time)
        writer_counter = save_data['writer_counter']

    writer = SummaryWriter(run_name)
    epoch_wp = 250
    EPOCHS = epoch_wp*num_wp

    agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name, epochs=EPOCHS, load_models=args.load_models)
    reset_probs = agent.reset_probs
    reset_model_nums = agent.reset_model_nums


    # Main loop
    total_steps = 0
    for i_episode in tqdm(range(args.start_epoch, EPOCHS)):
        if (i_episode + 1) % epoch_wp == 0:
            agent.save_model(save_name)
            # agent.eval_best_model(memory, save_name)
            wp_id += 1
            agent = OAT(state_dim=4, objs=objs, wp_id=wp_id, save_name=save_name, epochs=EPOCHS)
             # Memory
            memory = MyMemory()

        writer_counter = i_episode + 1
        i_episode = (i_episode % epoch_wp) + 1

        # if np.random.rand()<0.05 and i_episode<100 and i_episode>1:
        #     agent.reset_model(np.random.randint(10))

        if reset_probs[i_episode-1]<0.05 and i_episode<100 and i_episode>1:
            agent.reset_model(reset_model_nums[i_episode-1])


        # initialize variables
        episode_reward = 0
        done, truncated = False, False
        obs = env.reset()
        # print(obs['can_pos'])

        if args.env == 'Stack':
            objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
        elif args.env == 'Lift':
            objs = obs['cube_pos']
        elif args.env == 'PickPlace':
            objs = obs[args.object+'_pos']
        elif args.env == 'NutAssembly':
            nut = 'RoundNut'# if obs['nut_id'] == 0 else 'RoundNut'
            objs = obs[nut + '_pos']


        # select optimal trajectory
        traj_full = agent.traj_opt(i_episode, objs)
        
        traj = traj_full[-4:]
        
        traj_mat = np.reshape(traj_full, (wp_id,4))[:, :3] + obs['robot0_eef_pos']
        gripper_mat = np.reshape(traj_full, (wp_id, 4))[:, 3:] 

        # tqdm.write("traj = {}, {}".format(traj_mat, gripper_mat))


        if len(memory) > batch_size:
                for _ in range(100):
                    critic_loss = agent.update_parameters(memory, batch_size)
                    writer.add_scalar('model/critic', critic_loss, total_steps)
        
        time_s = 0
        train_reward = 0
        for timestep in range(wp_id*50):

            if args.render:
                env.render()    # toggle this when we don't want to render

            # convert traj to actions
            state = obs['robot0_eef_pos']
            g_state = obs['robot0_gripper_qpos']
            widx = int(np.floor(timestep / (50)))
            error = traj_mat[widx, :] - state
            
            if timestep < 10:
                full_action = np.array(list(10. * error) + [0.]*3 + [-1.])
            elif time_s >= 35:
                # full_action = np.array(list(10. * error) + [0.]*3 + list(gripper_mat[widx]))
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
        
        memory.push(np.concatenate((traj_full, objs)), episode_reward)
        save_data['episode'].append(i_episode)
        save_data['reward'].append(episode_reward)
        save_data['writer_counter'] = writer_counter

        
        if train_reward > agent.best_reward:
            agent.set_init(traj_full, train_reward)
            agent.save_model(save_name)
            save_data['best_traj'].append(episode_reward)
            pickle.dump(traj_full, open('models/' + save_name + '/traj.pkl', 'wb'))

        writer.add_scalar('reward', episode_reward, writer_counter)        
        pickle.dump(save_data, open('models/' + save_name + '/data.pkl', 'wb'))
        tqdm.write("wp_id: {}, Episode: {}, Reward_full: {}; Reward: {}, Predicted: {}".format(wp_id, writer_counter, round(episode_reward, 2), round(train_reward, 2), round(agent.get_avg_reward(traj_full), 2)))




if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True)
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')
    p.add_argument('--n_inits', type=int, default=1)
    p.add_argument('--num_wp', type=int, default=5)
    p.add_argument('--render', action='store_true', default=False)
    p.add_argument('--load_models', action='store_true', default=False)
    p.add_argument('--start_wp', type=int, default=1)
    p.add_argument('--start_epoch', type=int, default=0)
    p.add_argument('--run_time', type=str)
    args = p.parse_args()
    run_ours(args)

