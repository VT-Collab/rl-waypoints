import numpy as np
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import time
import os, sys

from memory import MyMemory
from method import Method


class train:
    def __init__(self, config):
        self.config = config
        self.env = config['task']['name']
        self.num_wp = config['num_wp']
        self.render = config['render']
        self.n_inits = config['n_inits']
        self.run_name = config['run_name']
        self.object = None if config['object']=='' else config['object']

        self.robot = config['task']['env']['robot']
        self.state_dim = config['task']['env']['state_dim']
        self.gripper_steps = config['task']['env']['gripper_steps']
        self.wp_steps = config['task']['env']['wp_steps']
        self.use_latch = config['task']['env']['use_latch']

        self.batch_size = config['task']['task']['batch_size']
        self.epoch_wp = config['task']['task']['epoch_wp']
        self.rand_reset_epoch = config['task']['task']['rand_reset_epoch']

        self.train()


    
    def reset_env(self, env, get_objs=False):
        obs = env.reset()
        if get_objs:
            if self.env == 'Lift':
                objs = obs['cube_pos']
            elif self.env == 'Stack':
                objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
            elif self.env == 'NutAssembly':
                nut = 'RoundNut'
                objs = obs[nut + '_pos']
            elif self.env == 'PickPlace':
                objs = obs[self.object+'_pos']
            elif self.env == 'Door':
                objs = np.concatenate((obs['door_pos'], obs['handle_pos']), axis=-1)
            return obs, objs
        return obs

    def get_state(self, obs):
        if self.env == 'Door':
            robot_pos = obs['robot0_eef_pos']
            robot_ang = R.from_quat(obs['robot0_eef_quat']).as_euler('xyz', degrees=False)
            state = np.concatenate((robot_pos, robot_ang), axis=-1)
            return state

        state = obs['robot0_eef_pos']
        return state

    def get_action(self, env, wp_idx, state, traj_mat, gripper_mat, time_s, timestep):
        error = traj_mat[wp_idx, :] - state
        if timestep < 10:
            full_action = np.array(list(10.*error) + [0.]*(6-len(state)) +[-1.])
        elif time_s >= self.wp_steps - self.gripper_steps:
            full_action = np.array([0.]*6 + list(gripper_mat[wp_idx]))
        else:
            full_action = np.array(list(10.*error)  +[0.]*(6-len(state)) + [0.])

        return full_action

    def train(self):
        save_data = {'episode': [], 'reward': []}

        if self.object is None:
            save_name = self.env + '/' + self.run_name
            if self.env == 'Door' and self.use_latch:
                save_name = self.env + '/with_latch/' + self.run_name
            elif self.env == 'Door' and not self.use_latch:
                save_name = self.env + 'without_latch/' + self.run_name
        else:
             save_name = self.env + '/' + self.object + '/' + self.run_name


        # save_dir = 'models/' + save_name
        # if not os.path.exists(save_dir):
        #     print("MAKING")
        #     os.makedirs(save_dir)

        controller_config = load_controller_config(default_controller="OSC_POSE")

        env = suite.make(
            env_name=self.env,
            robots=self.robot,
            controller_configs=controller_config,
            has_renderer=self.render,
            reward_shaping=True,
            control_freq=10,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            initialization_noise=None,
            single_object_mode=2,
            object_type=self.object,
            use_latch=self.use_latch,
        )

        wp_id = 1

        obs, objs = self.reset_env(env, get_objs=True)

        agent = Method(state_dim=self.state_dim, objs=objs, wp_id=wp_id, save_name=save_name, config=self.config)

        memory = MyMemory()

        run_name = 'runs/ours_' + self.run_name + datetime.datetime.now().strftime("%H-%M")
        writer = SummaryWriter(run_name)

        EPOCHS = self.epoch_wp * self.num_wp

        total_steps = 0
        for i_episode in tqdm(range(1, EPOCHS)):
            if i_episode % self.epoch_wp == 0:
                agent.save_model(save_name)

                wp_id += 1
                agent = Method(state_dim=self.state_dim, objs=objs, wp_id=wp_id, save_name=save_name, config=self.config)

                memory = MyMemory()

            i_episode = i_episode%self.epoch_wp

            if np.random.rand()<0.05 and i_episode<self.rand_reset_epoch and i_episode > 1:
                agent.reset_model(np.random.randint(10))

            episode_reward = 0
            done, truncated = False, False
            obs, objs = self.reset_env(env, get_objs=True)

            traj_full = agent.traj_opt(i_episode, objs)

            state = self.get_state(obs)
            traj_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, :self.state_dim-1] + state
            gripper_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, self.state_dim-1:]

            if len(memory) > self.batch_size:
                for _ in range(100):
                    critic_loss = agent.update_parameters(memory, self.batch_size)
                    writer.add_scalar('model/critic_loss', critic_loss, total_steps)

            time_s = 0
            train_reward = 0
            for timestep in range(wp_id*self.wp_steps):
                if self.render:
                    env.render()

                state = self.get_state(obs)
                wp_idx = timestep//50

                action = self.get_action(self.env, wp_idx, state, traj_mat, gripper_mat, time_s, timestep)

                time_s += 1
                if time_s >= 50:
                    time_s = 1

                obs, reward, done, _ = env.step(action)
                episode_reward += reward

                if timestep//50 == wp_id - 1:
                    train_reward += reward

                total_steps += 1

            memory.push(np.concatenate((traj_full, objs)), episode_reward)
            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)

            if train_reward > agent.best_reward:
                agent.set_init(traj_full, train_reward)
                agent.save_model(save_name)

            writer.add_scalar('reward', episode_reward, i_episode)
            tqdm.write("wp_id: {}, Episode: {}, Reward_full: {}; Reward: {}, Predicted: {}".format(wp_id, i_episode, round(episode_reward, 2), round(train_reward, 2), round(agent.get_avg_reward(traj_full), 2)))

            pickle.dump(save_data, open('models/' + save_name + '/data.pkl', 'wb'))

        exit()







