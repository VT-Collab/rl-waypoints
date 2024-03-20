import numpy as np
import gym
import datetime
import torch
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, LinearConstraint
import time
import os, sys
import random
from models import RNetwork


class Method(object):
    def __init__(self, state_dim, objs, wp_id, save_name, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        # self.objs = torch.FloatTensor(objs).to(device=self.device)
        self.wp_id = wp_id
        self.best_wp = []
        self.n_inits = 5
        self.lr = 1e-3
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.learned_models = []

        self.action_dim = config['task']['task']['action_space']

        save_dir = 'models/' + save_name

        for wp_id in range(1, self.wp_id+1):
            models = []
            for idx in range(self.n_models):
                critic = RNetwork(wp_id*self.state_dim + len(objs), hidden_dim=self.hidden_size).to(device=self.device)
                critic.load_state_dict(torch.load(save_dir + '/model_' + str(wp_id) + '_' + str(idx)))
                models.append(critic)
            self.learned_models.append(models)


        self.best_traj = self.action_dim*(np.random.rand(self.state_dim*self.wp_id) - 0.5)
        self.best_reward = -np.inf
        self.lincon = LinearConstraint(np.eye(self.state_dim), -self.action_dim, self.action_dim)

    def traj_opt(self, i_episode, objs):
        self.reward_idx = random.choice(range(self.n_models))
        self.objs = torch.FloatTensor(objs).to(device=self.device)
        self.curr_episode = i_episode

        self.traj = []
        for idx in range(1, self.wp_id+1):
            min_cost = np.inf

            self.load_model = True if idx!=self.wp_id else False
            self.curr_wp = idx-1

            for t_idx in range(self.n_inits):
                xi0 = np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim]) + np.random.normal(0, 0.1, size=self.state_dim) if t_idx!=0 else np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim])

                res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon, options={'eps': 1e-6, 'maxiter': 1e6})
                if res.fun < min_cost:
                    min_cost = res.fun
                    self.best_wp = res.x

            self.traj.append(self.best_wp)
        return np.array(self.traj).flatten()

    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)

    def get_cost(self, traj):
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj_learnt = torch.FloatTensor(np.array(self.traj).flatten()).to(device=self.device)
        traj = torch.cat((traj_learnt, traj)).to(device=self.device)
        traj = torch.cat((traj, self.objs)).to(device=self.device)
        reward = self.get_reward(traj)
        return -reward
    
    def get_reward(self, traj):
        loss = 0
        models = self.learned_models[self.curr_wp]
        for idx in range (self.n_models):
            critic = models[idx]
            loss += critic(traj).item()
        return loss/self.n_models


class evaluate:
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

        self.n_eval = config['task']['task']['num_eval']

        self.eval()


    
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

    def eval(self):
        save_data = {'episode': [], 'reward': []}

        if self.object is None:
            save_name = self.env + '/' + self.run_name
            if self.env == 'Door' and self.use_latch:
                save_name = self.env + '/with_latch/' + self.run_name
            elif self.env == 'Door' and not self.use_latch:
                save_name = self.env + 'without_latch/' + self.run_name
        else:
             save_name = self.env + '/' + self.object + '/' + self.run_name


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

        wp_id = self.num_wp

        obs, objs = self.reset_env(env, get_objs=True)

        agent = Method(state_dim=self.state_dim, objs=objs, wp_id=wp_id, save_name=save_name, config=self.config)


        EPOCHS = self.n_eval

        total_steps = 0
        for i_episode in tqdm(range(1, EPOCHS)):

            episode_reward = 0
            done, truncated = False, False
            obs, objs = self.reset_env(env, get_objs=True)

            traj_full = agent.traj_opt(i_episode, objs)

            state = self.get_state(obs)
            traj_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, :self.state_dim-1] + state
            gripper_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, self.state_dim-1:]

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

            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)


            pickle.dump(save_data, open('models/' + save_name + '/eval_data.pkl', 'wb'))

        exit()







