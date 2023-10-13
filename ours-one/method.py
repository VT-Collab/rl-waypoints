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
        self.models = []
        self.n_inits = 10

        # Reward Models
        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim*self.n_waypoints, self.hidden_size)
            optimizer = Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        # Trajectory Optimization
        self.best_reward = -np.inf
        self.best_traj = None


    # trajectory optimization over sampled reward function
    # hyperparameters: state space limits in linear constraint, noise in xi0
    def traj_opt(self, widx):
        self.widx = widx
        xi_star, cost_min = None, np.inf
        xi_len = self.widx * self.state_dim
        lin_con = LinearConstraint(np.eye(xi_len), -0.5, 0.5)
        self.reward_idx = np.random.choice(self.n_models)
        for idx in range(self.n_inits):
            xi0 = 0.5*(np.random.rand(xi_len)-0.5)
            res = minimize(self.get_cost, xi0, 
                            method='SLSQP', 
                            constraints=lin_con, 
                            options={'eps': 1e-6, 'maxiter': 1e6})
            if res.fun < cost_min:
                cost_min = res.fun
                xi_star = res.x
        # exploration of the gripper
        xi_star[-1] = np.random.choice([-1.0, 1.0])
        print(xi_star)
        return xi_star


    # get cost for trajectory optimizer
    def get_cost(self, traj):
        xi = np.reshape(traj, (-1, self.state_dim))
        xi_full = np.zeros((self.n_waypoints, self.state_dim))
        xi_full[:self.widx,:] = xi
        xi_full[self.widx:,:] = xi[-1,:]
        traj_full = np.reshape(xi_full, (-1,))
        traj_full = torch.FloatTensor(traj_full)
        return -self.get_reward(traj_full, self.reward_idx)


    # reset a reward model
    def reset_model(self, index):
        critic = RNetwork(self.state_dim*self.n_waypoints, self.hidden_size)
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