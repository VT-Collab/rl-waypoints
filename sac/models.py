import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)


    # map state + action to two Q-values
    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        # Q-value 1
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = 1000*self.linear3(x1)
        # Q-value 2
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = 1000*self.linear6(x2)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    # map state to mean + log_std over action
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    # sample a state from Gaussian distribution
    def sample(self, state):
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * 0.2
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(0.2 * (1 - y_t.pow(2)) + epsilon)
        mean = torch.tanh(mean) * 0.2
        return action, log_prob, mean