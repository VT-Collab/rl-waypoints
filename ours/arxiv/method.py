import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.convert_parameters import parameters_to_vector
from models import RNetwork
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import os, sys
import random
from tqdm import tqdm

class Method(object):
    def __init__(self, state_dim, objs, save_name):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters   
        self.traj_dim = state_dim
        self.state_dim = state_dim
        self.lr = 1e-3
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.n_inits = 5
        self.objs = torch.FloatTensor(objs).to(device=self.device)


        # Critic
        for _ in range(self.n_models):
            critic = RNetwork(self.traj_dim + len(self.objs), hidden_dim=self.hidden_size).to(device=self.device)
            optimizer = Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        # Actor
        self.best_reward = -np.inf
        self.best_traj = 0.5*(np.random.rand(self.traj_dim)-0.5)
        self.lin_con = LinearConstraint(np.eye(self.traj_dim), -0.5, 0.5)
        self.reward_idx = None


    # trajectory optimization over sampled reward function
    def traj_opt(self, episode, objs):

        self.reward_idx = random.choice(range(self.n_models))
        self.objs = torch.FloatTensor(objs).to(device=self.device)
        traj_star, min_cost = None, np.inf

        if episode < 50 or np.random.rand() < 25/episode:
            traj_star = 0.5*(np.random.rand(self.state_dim) - 0.5)
        else:
            for idx in range(self.n_inits):
                xi0 = np.copy(self.best_traj) + np.random.normal(0, 0.1, size=self.traj_dim) if idx!=0 else np.copy(self.best_traj)
                res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lin_con, options={'eps': 1e-6, 'maxiter': 1e6})
                if res.fun < min_cost:
                    min_cost = res.fun
                    traj_star = res.x
        return traj_star


    # set the initial parameters for trajectory optimization
    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)


    # get cost for trajectory optimizer
    def get_cost(self, traj):
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj = torch.cat((traj, self.objs)).to(device=self.device)
        reward = self.get_reward(traj, self.reward_idx)
        return -reward


    # get average and std reward across all models
    def get_avg_reward(self, traj):
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj = torch.cat((traj, self.objs))
        R = np.zeros((self.n_models,))
        for idx in range(self.n_models):
            R[idx] = self.get_reward(traj, idx)
        return np.mean(R)


    # get reward from specific model
    def get_reward(self, traj, reward_idx):
        critic, _ = self.models[reward_idx]
        return critic(traj).item()


    # train all the reward models
    def update_parameters(self, memory, batch_size):
        loss = np.zeros((self.n_models,))
        for idx, (critic, optimizer) in enumerate(self.models):
            loss[idx] = self.update_critic(critic, optimizer, memory, batch_size)
        return np.mean(loss)


    # train a specific reward model
    def update_critic(self, critic, optimizer, memory, batch_size):

        # sample a batch of (trajectory, reward) from memory
        trajs, rewards = memory.sample(batch_size)
        trajs = torch.FloatTensor(trajs).to(device=self.device)
        

        rewards = torch.FloatTensor(rewards).to(device=self.device).unsqueeze(1)
        
        # train critic using supervised approach
        rhat= critic(trajs).to(device=self.device)
        # q_loss = F.mse_loss(rhat, rewards)
        loss1 = F.binary_cross_entropy(torch.sigmoid(rhat), rewards)
        loss2 = 0.005*parameters_to_vector(critic.parameters()).norm() ** 2
        q_loss = loss1 + loss2
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        return q_loss.item()

    def reset_model(self, idx):
        critic = RNetwork(self.traj_dim + len(self.objs), hidden_dim=self.hidden_size).to(device=self.device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr) 
        self.models[idx] = (critic, optimizer)
        tqdm.write("RESET MODEL {}".format(idx))

    def save_model(self, save_name:str):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, (critic, optimizer) in enumerate(self.models):
            torch.save(critic.state_dict(), save_dir + '/model_' + str(idx))