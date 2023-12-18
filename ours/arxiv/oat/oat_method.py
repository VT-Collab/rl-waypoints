import numpy as np
import pickle
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, LinearConstraint
import os, sys
import random
from models import RNetwork, GRU

class OAT(object):
    def __init__(self, state_dim, objs, wp_id):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.objs = torch.FloatTensor(objs).to(device=self.device)
        self.wp_id = wp_id
        self.best_wp = []
        self.n_inits = 1
        self.lr = 1e-3
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.n_eval = 100

        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim + len(self.objs), hidden_dim=self.hidden_size).to(device=self.device)
            optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        self.best_traj = 0.5*(np.random.rand(self.state_dim) - 0.5)
        self.best_reward = -np.inf
        self.lincon = LinearConstraint(np.eye(self.state_dim), -0.25, 0.25)

    def traj_opt(self, episode):
        self.reward_idx = random.choice(range(self.n_models))
        min_cost = np.inf
        traj_star = None

        for idx in range(self.n_inits):
            xi0 = np.copy(self.best_traj[-4:]) + np.random.normal(0, 0.1, size=self.state_dim) if idx!=0 else np.copy(self.best_traj[-4:])
            res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon, options={'eps': 1e-6, 'maxiter': 1e6})
            if res.fun < min_cost:
                min_cost = res.fun
                if len(self.best_wp) < 1:
                    self.best_wp.append(res.x)
                else:
                    self.best_wp[-1] = res.x
            return np.array(self.best_wp).flatten()

    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)

    def get_cost(self, traj):
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj = torch.cat((traj, self.objs)).to(device=self.device)
        reward = self.get_reward(traj, self.reward_idx)
        return -reward
    
    def get_reward(self, traj, reward_idx):
        loss = 0
        critic, _ = self.models[reward_idx]
        loss += critic(traj).item()
        return loss

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
        optim.zero_grad
        loss.backward()
        optim.step()
        return loss.item()


    def save_model(self, save_name:str):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, (critic, optimizer) in enumerate(self.models):
            torch.save(critic.state_dict(), save_dir + '/model_' + str(self.wp_id) + '_' + str(idx))



"""
PSEUDOCODE

Initialize start state of robot x_s
Set max number of waypoints N_WP
traj = []

for waypoint_id in range (1, N_WP); do
Initialize reward model R with weights theta_waypoint_id
Initialize Memory
Initialize best_wp

    for N in range (1, EPOCHS); do
        get the current state x of the robot
        select action using trajopt and R
        get true reward R_t
        Memory -> (x, R_t)
        compute loss (R, R_t)
        Update theta_1
        update best_wp 
    end for
    traj -> best_wp
    x_s -> best_wp

end for
"""



"""
TO DO:
Check by changing the network to a waypoint predictor from a reward predictor
"""