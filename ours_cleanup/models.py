import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class WeightClipper(object):

    def __init__(self, frequency=5):
        # self.frequency = frequency
        pass
    
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0, np.inf)
            module.weight.data = w

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RNetwork(nn.Module):
    def __init__(self, traj_dim, hidden_dim):
        super(RNetwork, self).__init__()
        self.ReLU = nn.LeakyReLU()
        self.ELU = nn.ELU()
        self.linear1 = nn.Linear(traj_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, traj):
        x = self.ReLU(self.linear1(traj))
        x = self.ReLU(self.linear2(x))
        return self.linear3(x)


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.actor(x)

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRU, self).__init__()
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        x, h = self.gru(x, h)
        x = self.linear1(self.relu(x[:, -1]))
        return self.relu(x), h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(AE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, int(self.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim/2), self.output_dim),
            # nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, int(self.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim/2), 1)
        )

    def forward(self, x):
        wp = self.encoder(x)
        # traj = torch.cat(wp, x[self.output_dim:])
        reward = self.decoder(wp)
        return wp, reward