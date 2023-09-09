import torch
import torch.nn.functional as F
from torch.optim import Adam
from models import RNetwork
import numpy as np
from scipy.optimize import minimize, LinearConstraint


class Method(object):
    def __init__(self, traj_dim):

        # hyperparameters   
        self.traj_dim = traj_dim
        self.lr = 0.001
        self.hidden_size = 64
        self.n_models = 30
        self.n_samples = 5
        self.models = []
        self.n_inits = 5

        # Critic
        for _ in range(self.n_models):
            critic = RNetwork(self.traj_dim, hidden_dim=self.hidden_size)
            optimizer = Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        # Actor
        self.best_reward = -np.inf
        self.best_traj = 0.5*(np.random.rand(self.traj_dim)-0.5)
        self.lin_con = LinearConstraint(np.eye(self.traj_dim), -0.5, 0.5)
        self.reward_idx = None


    # trajectory optimization over sampled reward function
    def traj_opt(self):
        xi_star, min_cost = None, np.inf
        self.reward_idx = np.random.choice(self.n_models, self.n_samples, replace=True)
        for idx in range(self.n_inits):
            if idx < 1:
                xi0 = np.copy(self.best_traj)
            else:
                xi0 = np.copy(self.best_traj) + np.random.normal(0, 0.2, size=self.traj_dim)
            res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lin_con, options={'eps': 1e-6, 'maxiter': 1e6})
            if res.fun < min_cost:
                min_cost = res.fun
                xi_star = res.x
        return xi_star


    # set the initial parameters for trajectory optimization
    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)


    # get cost for trajectory optimizer
    def get_cost(self, traj):
        traj = torch.FloatTensor(traj)
        reward = 0
        for idx in self.reward_idx:
            reward += self.get_reward(traj, idx)
        return -reward / (1.0 * len(self.reward_idx))


    # get average and std reward across all models
    def get_avg_reward(self, traj):
        R = np.zeros((self.n_models,))
        for idx in range(self.n_models):
            R[idx] = self.get_reward(traj, idx)
        return np.mean(R), np.std(R), R


    # get reward from specific model
    def get_reward(self, traj, idx):
        critic, _ = self.models[idx]
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
        trajs = torch.FloatTensor(trajs)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        # train critic using supervised approach
        rhat = critic(trajs)
        q_loss = F.mse_loss(rhat, rewards)
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        return q_loss.item()