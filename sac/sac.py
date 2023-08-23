import torch
import torch.nn.functional as F
from torch.optim import Adam
from models import QNetwork, GaussianPolicy
import numpy as np


class SAC(object):
    def __init__(self, state_dim, action_dim):

        # hyperparameters     
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 0.0003
        self.hidden_size = 32

        # Critic
        self.critic = QNetwork(state_dim, action_dim, hidden_dim=self.hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim=self.hidden_size)
        self.hard_update(self.critic_target, self.critic)

        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim=self.hidden_size)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)


    # use the policy to select an action
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action, _, _ = self.policy.sample(state)
        return action.detach().numpy()


    # train the Q-functions
    def update_parameters(self, memory, batch_size):

        # sample a batch from memory
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # train critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi.sum(dim=1).unsqueeze(1)
            next_q_value = rewards + (1-dones) * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # train actor
        pi, log_pi, _ = self.policy.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # update target network
        self.soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()


    # helper functions for updating the weights of the Qfunctions
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
