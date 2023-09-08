import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RNetwork(nn.Module):
    def __init__(self, traj_dim, hidden_dim):
        super(RNetwork, self).__init__()
        self.ReLU = nn.LeakyReLU()
        self.linear1 = nn.Linear(traj_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, traj):
        x = self.ReLU(self.linear1(traj))
        x = self.ReLU(self.linear2(x))
        return self.linear3(x)