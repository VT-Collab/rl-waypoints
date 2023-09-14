import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(RNetwork, self).__init__()

        self.rnn = nn.GRU(
                    input_size=state_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True
                    )

        self.ReLU = nn.LeakyReLU()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, traj):
        _, h = self.rnn(traj)
        x = self.ReLU(self.linear1(h))
        return self.linear2(x)

    def segment_rewards(self, traj):
        z, _ = self.rnn(traj)
        x = self.ReLU(self.linear1(z))
        return self.linear2(x)