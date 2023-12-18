import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from models import RNetwork, Actor
import os, sys
import random


class Method(object):
    def __init__(self, traj_dim, env_dim):
        self.traj_dim = traj_dim
        self.env_dim = env_dim
        self.hidden_dim = 128
        self.lr = 1e-3
        self.n_models = 10
        self.models = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        for _ in range(self.n_models):
            critic = RNetwork(self.traj_dim, self.hidden_dim).to(self.device)
            optimizer = optim.Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))

        self.actor = Actor(self.env_dim, self.hidden_dim, self.traj_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.best_reward = -np.inf
        self.best_traj = 0.5*(np.random.rand(self.traj_dim) - 0.5)
        self.reward_idx = None

    
    def traj_opt(self, objs):
        objs = torch.FloatTensor(objs).to(self.device)
        traj = self.actor(objs)
        return traj.cpu().detach().numpy()
    
    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = traj

    def update_parameters(self, memory, batch_size):
        critic_loss = np.zeros((self.n_models,))
        actor_loss = np.zeros((self.n_models,))
        for idx, (critic, optimizer) in enumerate(self.models):
            critic_loss[idx] = self.update_critic(critic, optimizer, memory, batch_size)
            actor_loss[idx] = self.update_actor(self.actor, self.actor_optim, critic, memory, batch_size)
        return np.mean(critic_loss), np.mean(actor_loss)


    # train a specific reward model
    def update_critic(self, critic, optimizer, memory, batch_size):

        # sample a batch of (trajectory, reward) from memory
        trajs, _, rewards = memory.sample(batch_size)
        trajs = torch.FloatTensor(trajs).to(device=self.device)
        rewards = torch.FloatTensor(rewards).to(device=self.device).unsqueeze(1)
        
        # train critic using supervised approach
        rhat = critic(trajs).to(device=self.device)
        q_loss = F.mse_loss(rhat, rewards)
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        return q_loss.item()

    def update_actor(self, actor, optimizer, critic, memory, batch_size):
        trajs, objs, rewards = memory.sample(batch_size)
        trajs = torch.FloatTensor(trajs).to(self.device)
        objs = torch.FloatTensor(objs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        trajs = actor(objs).to(device=self.device)
        # traj_len = 0
        # for idx in range(len(trajs) - 1):
        #     traj_len += torch.norm(trajs[idx+1] - trajs[idx])
        rhat = critic(trajs).to(device=self.device)
        loss = F.mse_loss(rewards, rhat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def save_models(self, save_name):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, (critic, optimizer) in enumerate(self.models):
            torch.save(critic.state_dict(), save_dir + '/critic_' + str(idx))
        torch.save(self.actor.state_dict(), save_dir + '/actor')