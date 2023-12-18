import numpy as np
from memory import MyMemory
from method import Method
import gym 
import time
import pickle
import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.controllers import load_controller_config

class TRAIN():
    def __init__(self, args):
        self.save_data = {'episode': [], 'reward': []}

        if args.object == 'test':
            self.save_name = args.env + '/' + args.run_num 
        else:
            self.save_name = args.env + '/' + args.object + '/' + args.run_num 
        self.batch_size = 30

        # load default controller parameters for Operational Space Control (OSC)
        controller_config = load_controller_config(default_controller="OSC_POSE")

        # create environment instance
        self.env = suite.make(
            env_name=args.env, # try with other tasks like "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            controller_configs=controller_config,
            has_renderer=args.render,
            reward_shaping=False,
            control_freq=10,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            initialization_noise=None,
            single_object_mode=2,
            object_type=args.object,
            use_latch=False,
        )

        self.train(args)

    def train(self, args):
        num_wp = args.num_wp
        obs = self.env.reset()
        if args.env == 'Stack':
            objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
        elif args.env == 'Lift':
            objs = obs['cube_pos']
        elif args.env == 'PickPlace':
            objs = obs[args.object+'_pos']
        
        agent = Method(state_dim=4*num_wp, objs=objs, save_name=self.save_name)

        memory = MyMemory(args)

        EPOCHS = 1000
        total_steps = 0

        for i_episode in tqdm(range(1, EPOCHS)):
            if i_episode%100 == 0:
                agent.save_model(self.save_name)

            if np.random.rand() < 0.05 and i_episode < 900 and i_episode > 100:
                agent.reset_model(np.random.randint(10))

            episode_reward = 0
            done, truncated = False, False

            obs = self.env.reset()
            if args.env == 'Stack':
                objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
            elif args.env == 'Lift':
                objs = obs['cube_pos']
            elif args.env == 'PickPlace':
                objs = obs[args.object+'_pos']

            traj_full = agent.traj_opt(i_episode, objs)

            traj_mat = np.reshape(traj_full, (num_wp,4))[:, :3] + obs['robot0_eef_pos']
            gripper_mat = np.reshape(traj_full, (num_wp, 4))[:, 3:]

            if i_episode > self.batch_size:
                for _ in range (100):
                    ciritc_loss = agent.update_parameters(memory, self.batch_size)
                
            time_s = 0
            train_reward = 0
            for timestep in range(num_wp*50):
                if args.render:
                    self.env.render()
                
                state = obs['robot0_eef_pos']
                g_state = obs['robot0_gripper_qpos']
                widx = int(np.floor(timestep/50))
                error = traj_mat[widx, :] - state

                if timestep < 10:
                    full_action = np.array(list(10. * error) + [0]*3 + [-1.])
                elif time_s >= 35:
                    full_action = np.array(list(10. * error) + [0.]*3 + list(gripper_mat[widx]))
                else: 
                    full_action = np.array(list(10. * error) + [0]*4)

                time_s += 1
                if time_s >= 50:
                    time_s = 1

                obs, reward, done, _ = self.env.step(full_action)
                episode_reward = 1. if reward > 2 else 0.
                total_steps += 1

            memory.push(np.concatenate((traj_full, objs)), episode_reward)
            self.save_data['episode'].append(i_episode)
            self.save_data['reward'].append(episode_reward)

            if episode_reward > agent.best_reward:
                agent.set_init(traj_full, episode_reward)
                agent.save_model(self.save_name)

            tqdm.write("Episode: {}, Reward_full: {}, Predicted: {}".format(i_episode, round(episode_reward, 2), round(agent.get_avg_reward(traj_full), 2)))

        pickle.dump(self.save_data, open('models/' + self.save_name + '/data.pkl', 'wb'))



if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True)
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')
    p.add_argument('--n_inits', type=int, default=1)
    p.add_argument('--num_wp', type=int, default=5)
    p.add_argument('--render', action='store_true', default=False)
    args = p.parse_args()
    TRAIN(args)
