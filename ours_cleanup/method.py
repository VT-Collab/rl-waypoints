import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, LinearConstraint
import os
import random
from models import RNetwork
from tqdm import tqdm

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
        self.n_eval = 100

        self.action_dim = config['task']['task']['action_space']
        self.exploration_epoch = config['task']['task']['exploration_epoch']
        self.ensemble_sampling_epoch = config['task']['task']['ensemble_sampling_epoch']
        self.averaging_noise_epoch = config['task']['task']['averaging_noise_epoch']

        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim*(self.wp_id) + len(objs), hidden_dim=self.hidden_size).to(device=self.device)
            optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        save_dir = 'models/' + save_name

        for wp_id in range(1, self.wp_id):
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

            if idx == self.wp_id and i_episode <= self.exploration_epoch:
                self.best_wp = self.action_dim*(np.random.rand(self.state_dim) - 0.5)

            else:
                for t_idx in range(self.n_inits):
                    xi0 = np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim]) + np.random.normal(0, 0.1, size=self.state_dim) if t_idx!=0 else np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim])

                    res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon, options={'eps': 1e-6, 'maxiter': 1e6})
                    if res.fun < min_cost:
                        min_cost = res.fun
                        self.best_wp = res.x
                if idx == self.wp_id and np.random.rand() < 0.5 and i_episode>self.ensemble_sampling_epoch and i_episode<self.averaging_noise_epoch:
                    tqdm.write("NOISE ADDED TO GRIPPER")
                    self.best_wp[-1] *= -1
                if idx == self.wp_id and np.random.rand() < 0.5 and i_episode>self.ensemble_sampling_epoch and i_episode<self.averaging_noise_epoch:
                    tqdm.write("NOISE ADDED TO POSE")
                    self.best_wp[:3] += np.random.normal(0, 0.05, 3)

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
        if self.load_model:
            models = self.learned_models[self.curr_wp]
            for idx in range (self.n_models):
                critic = models[idx]
                loss += critic(traj).item()
            return loss/self.n_models

        else:
            if self.curr_episode < self.ensemble_sampling_epoch :
                critic, _ = self.models[self.reward_idx]
                return critic(traj).item()
            else:
                for idx in range(self.n_models):
                    critic, _ = self.models[idx]
                    loss += critic(traj).item()
                return loss/self.n_models

    def get_avg_reward(self, traj):
        reward = 0
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj = torch.cat((traj, self.objs))
        for idx in range(self.n_models):
            critic, _ = self.models[idx]
            reward += critic(traj).item()
        return reward/self.n_models

    def update_parameters(self, memory, batch_size):
        loss = np.zeros((self.n_models))
        for idx, (critic, optim) in enumerate(self.models):
            loss[idx] = self.update_critic(critic, optim, memory, batch_size)
        return np.mean(loss)

    def update_critic(self, critic, optim, memory, batch_size):
        trajs, rewards = memory.sample(batch_size)
        trajs = torch.FloatTensor(trajs).to(device=self.device)

        rewards = torch.FloatTensor(rewards).to(device=self.device).unsqueeze(1)

        rhat = critic(trajs).to(device=self.device)
        loss = F.mse_loss(rhat, rewards)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss.item()


    def reset_model(self, idx):
        critic = RNetwork(self.wp_id*self.state_dim + len(self.objs), hidden_dim=self.hidden_size).to(device=self.device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr) 
        self.models[idx] = (critic, optimizer)
        tqdm.write("RESET MODEL {}".format(idx))

    def save_model(self, save_name:str):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, (critic, optimizer) in enumerate(self.models):
            torch.save(critic.state_dict(), save_dir + '/model_' + str(self.wp_id) + '_' + str(idx))

