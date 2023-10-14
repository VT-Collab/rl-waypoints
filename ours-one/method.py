import torch
import torch.nn.functional as F
from torch.optim import Adam
from models import RNetwork
import numpy as np
from scipy.optimize import minimize, LinearConstraint



class Method(object):
    def __init__(self, state_dim, obs_dim):

        # hyperparameters
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.lr = 0.0003
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.n_inits = 10

        # Reward Models
        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim + self.obs_dim, self.hidden_size)
            optimizer = Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        # Trajectory Optimization
        self.best_reward = -np.inf
        self.best_traj = None
        self.lb = np.array([-0.25, -0.25, -0.25, -np.pi/4, -1.0])
        self.ub = np.array([0.25, 0.25, 0.25, np.pi/4, 1.0])
        self.lin_con = LinearConstraint(np.eye(self.state_dim), self.lb, self.ub)


    # sample a random waypoint
    def sample_waypoint(self):
        return np.random.uniform(self.lb, self.ub)


    # trajectory optimization over sampled reward function
    def traj_opt(self, obs, n_samples=1):
        self.obs = obs
        self.n_samples = n_samples
        xi_star, cost_min = None, np.inf
        self.reward_idx = np.random.choice(self.n_models, 
                                    self.n_samples, replace=False)
        for idx in range(self.n_inits):
            xi0 = self.sample_waypoint()
            res = minimize(self.get_cost, xi0, 
                            method='SLSQP', 
                            constraints=self.lin_con, 
                            options={'eps': 1e-6, 'maxiter': 1e6})
            if res.fun < cost_min:
                cost_min = res.fun
                xi_star = res.x
        return xi_star


    # get cost for trajectory optimizer
    def get_cost(self, traj):
        traj1 = np.concatenate((traj, self.obs), -1)
        traj1 = torch.FloatTensor(traj1)
        reward = np.zeros((self.n_samples,))
        for x, idx in enumerate(self.reward_idx):
            reward[x] = self.get_reward(traj1, idx)
        return -np.mean(reward)


    # reset a reward model
    def reset_model(self, index):
        critic = RNetwork(self.state_dim + self.obs_dim, self.hidden_size)
        optimizer = Adam(critic.parameters(), lr=self.lr)
        self.models[index] = (critic, optimizer)
        print("just reset model number: ", index)


    # get reward from specific model
    def get_reward(self, traj, idx):
        critic, _ = self.models[idx]
        return critic(traj).item()


    # get average reward across all models
    def get_avg_reward(self, traj):
        traj = torch.FloatTensor(traj)
        reward = np.zeros((self.n_models,))
        for idx in range(self.n_models):
            reward[idx] = self.get_reward(traj, idx)
        return np.mean(reward)


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