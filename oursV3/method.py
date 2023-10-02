import torch
import torch.nn.functional as F
from torch.optim import Adam
from models import RNetwork
import numpy as np
from scipy.optimize import minimize, LinearConstraint


class Method(object):
    def __init__(self, state_dim, n_waypoints):

        # hyperparameters
        self.state_dim = state_dim
        self.n_waypoints = n_waypoints
        self.lr = 0.0003
        self.hidden_size = 256
        self.n_models = 10
        self.n_samples = 1
        self.models = []
        self.n_inits = 1

        # Reward Models
        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim*self.n_waypoints, hidden_dim=self.hidden_size)
            optimizer = Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        # Trajectory Optimization
        self.best_reward = -np.inf
        self.best_traj = np.zeros((self.state_dim*self.n_waypoints,))
        self.lin_con = LinearConstraint(np.eye(self.n_waypoints*self.state_dim), -0.5, 0.5)


    # trajectory optimization over sampled reward function
    # hyperparameters: state space limits in linear constraint, noise in xi0
    def traj_opt(self):
        xi_star, cost_min = None, np.inf
        self.reward_idx = np.random.choice(self.n_models, self.n_samples, replace=True)
        for idx in range(self.n_inits):
            xi0 = np.copy(self.best_traj)
            # if idx > 0:
            xi0 += np.random.normal(0, 0.05, size=len(self.best_traj))
            res = minimize(self.get_cost, xi0, 
                            method='SLSQP', 
                            constraints=self.lin_con, 
                            options={'eps': 1e-6, 'maxiter': 1e6})
            if res.fun < cost_min:
                cost_min = res.fun
                xi_star = res.x
        return xi_star


    # reset a reward model
    def reset_model(self, index):
        critic = RNetwork(self.state_dim*self.n_waypoints, hidden_dim=self.hidden_size)
        optimizer = Adam(critic.parameters(), lr=self.lr)
        self.models[index] = (critic, optimizer)
        print("just reset model number: ", index)


    # set the initial parameters for trajectory optimization
    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)
        print("best_traj is now set with reward: ", reward)


    # set the number of samples we should average over
    def set_n_samples(self, n_samples):
        self.n_samples = n_samples
        print("n_samples is now set to: ", n_samples)


    # get reward from specific model
    def get_reward(self, traj, idx):
        critic, _ = self.models[idx]
        return critic(traj).item()


    # get cost for trajectory optimizer
    def get_cost(self, traj):
        traj = torch.FloatTensor(traj)
        reward = np.zeros((self.n_samples,))
        for x, idx in enumerate(self.reward_idx):
            reward[x] = self.get_reward(traj, idx)
        return -np.mean(reward)


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