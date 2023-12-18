# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from models import RNetwork
# import numpy as np
# from scipy.optimize import minimize, LinearConstraint
# import os, sys
# import random

# class Method(object):
#     def __init__(self, traj_dim, state_dim, n_traji, objs):
        
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # hyperparameters   
#         self.traj_dim = traj_dim
#         self.state_dim = state_dim
#         self.lr = 0.001
#         self.hidden_size = 256
#         self.n_models = 10
#         self.n_avg = 1
#         self.models = []
#         self.n_inits = n_traji
#         self.objs = torch.FloatTensor(objs).to(device=self.device)


#         print(self.device)

#         # Critic
#         for _ in range(self.n_models):
#             critic = RNetwork(self.traj_dim + len(self.objs), hidden_dim=self.hidden_size).to(device=self.device)
#             optimizer = Adam(critic.parameters(), lr=self.lr)
#             self.models.append((critic, optimizer))

#         # Actor
#         self.best_reward = -np.inf
#         self.best_traj = 0.5*(np.random.rand(self.traj_dim)-0.5)
#         self.lin_con = LinearConstraint(np.eye(self.traj_dim), -0.5, 0.5)
#         self.reward_idx = None
#         self.best_model = None
#         self.n_eval = 50

       


#     # trajectory optimization over sampled reward function
#     def traj_opt(self, episode):
#         traj_star, min_cost = None, np.inf
#         if self.best_model is not None:
#             if np.random.rand() > 0.2:
#                 self.reward_idx = self.best_model
#             else: 
#                 self.reward_idx = np.random.randint(self.n_models)
#         else:
#             self.reward_idx = np.random.randint(self.n_models)
#         # self.reward_idx = np.random.randint(self.n_models)
#         # self.reward_idx = random.sample(range(self.n_models), self.n_avg)

#         for idx in range(self.n_inits):
#             xi0 = np.copy(self.best_traj) + np.random.normal(0, 0.1, size=self.traj_dim) if idx!=0 else np.copy(self.best_traj)
#             res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lin_con, options={'eps': 1e-6, 'maxiter': 1e6})
#             if res.fun < min_cost:
#                 min_cost = res.fun
#                 traj_star = res.x
#         return traj_star


#     # set the initial parameters for trajectory optimization
#     def set_init(self, traj, reward):
#         self.best_reward = reward
#         self.best_traj = np.copy(traj)


#     # get cost for trajectory optimizer
#     def get_cost(self, traj):
#         traj = torch.FloatTensor(traj).to(device=self.device)
#         traj = torch.cat((traj, self.objs)).to(device=self.device)
#         reward = self.get_reward(traj, self.reward_idx)
#         return -reward


#     # get average and std reward across all models
#     def get_avg_reward(self, traj):
#         R = np.zeros((self.n_models,))
#         for idx in range(self.n_models):
#             R[idx] = self.get_reward(traj, idx)
#         return np.mean(R), np.std(R), R


#     # get reward from specific model
#     def get_reward(self, traj, reward_idx):
#         # reward = 0
#         # for idx in reward_idx:
#         #     critic, _ = self.models[idx]
#         #     reward += critic(traj).item() 
#         # return reward/len(self.reward_idx)
#         critic, _ = self.models[reward_idx]
#         return critic(traj).item()


#     # train all the reward models
#     def update_parameters(self, memory, batch_size):
#         loss = np.zeros((self.n_models,))
#         for idx, (critic, optimizer) in enumerate(self.models):
#             loss[idx] = self.update_critic(critic, optimizer, memory, batch_size)
#         return np.mean(loss)


#     # train a specific reward model
#     def update_critic(self, critic, optimizer, memory, batch_size):

#         # sample a batch of (trajectory, reward) from memory
#         trajs, rewards = memory.sample(batch_size)
#         trajs = torch.FloatTensor(trajs).to(device=self.device)
#         rewards = torch.FloatTensor(rewards).to(device=self.device).unsqueeze(1)
        
#         # train critic using supervised approach
#         rhat = critic(trajs).to(device=self.device)
#         q_loss = F.mse_loss(rhat, rewards)
#         optimizer.zero_grad()
#         q_loss.backward()
#         optimizer.step()
#         return q_loss.item()

#     def eval_best_model(self, memory, save_name:str):
#         loss_arr = np.zeros(self.n_models)
#         trajs, rewards = memory.sample(self.n_eval)
#         trajs = torch.FloatTensor(trajs).to(device=self.device)
#         rewards = torch.FloatTensor(rewards).to(device=self.device).squeeze()
#         for idx, (critic, optimizer) in enumerate(self.models):
#             rhat = critic(trajs).to(device=self.device).squeeze()
#             loss_arr[idx] = F.mse_loss(rhat, rewards).cpu().detach().numpy()

#         self.best_model = np.argmin(loss_arr)

#         save_dir = 'models/' + save_name
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
        
#         critic, _ = self.models[self.best_model]
#         torch.save(critic.state_dict(), save_dir + '/model_best')
        


#     def save_model(self, save_name:str):
#         save_dir = 'models/' + save_name
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
        
#         for idx, (critic, optimizer) in enumerate(self.models):
#             torch.save(critic.state_dict(), save_dir + '/model_' + str(idx))


import torch
import torch.nn.functional as F
from torch.optim import Adam
from models import RNetwork, GRU
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import os, sys
import random

class Method(object):
    def __init__(self, traj_dim, state_dim, n_traji, objs):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters   
        self.traj_dim = traj_dim
        self.state_dim = state_dim
        self.lr = 1e-3
        self.hidden_size = 256
        self.n_models = 10
        self.n_avg = 5
        self.models = []
        self.n_inits = n_traji
        self.objs = torch.FloatTensor(objs).to(device=self.device)


        # Critic
        for _ in range(self.n_models):
            # critic = GRU(input_dim, hidden_dim=128, output_dim=1, num_layers=2)
            critic = RNetwork(self.traj_dim + len(self.objs), hidden_dim=self.hidden_size).to(device=self.device)
            optimizer = Adam(critic.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 1000], gamma=1.0)
            self.models.append((critic, optimizer, scheduler))

        # Actor
        self.best_reward = -np.inf
        self.best_traj = 0.5*(np.random.rand(self.traj_dim)-0.5)
        self.lin_con = LinearConstraint(np.eye(self.traj_dim), -0.5, 0.5)
        self.reward_idx = None

       


    # trajectory optimization over sampled reward function
    def traj_opt(self, episode):
        # if episode%250 == 0:
        #     self.n_avg += 1
        traj_star, min_cost = None, np.inf
        # self.reward_idx = np.random.randint(self.n_models)
        self.reward_idx = random.sample(range(self.n_models), self.n_avg)

        for idx in range(self.n_inits):
            xi0 = np.copy(self.best_traj) + np.random.normal(0, 0.1, size=self.traj_dim) if idx!=0 else np.copy(self.best_traj)
            res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lin_con, options={'eps': 1e-6, 'maxiter': 1e6})
            if res.fun < min_cost:
                min_cost = res.fun
                traj_star = res.x
        return traj_star


    # set the initial parameters for trajectory optimization
    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)


    # get cost for trajectory optimizer
    def get_cost(self, traj):
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj = torch.cat((traj, self.objs)).to(device=self.device)
        reward, std = self.get_reward(traj, self.reward_idx)
        return -reward #- 0.1*np.sqrt(std)


    # get average and std reward across all models
    def get_avg_reward(self, traj):
        R = np.zeros((self.n_models,))
        for idx in range(self.n_models):
            R[idx] = self.get_reward(traj, idx)
        return np.mean(R), np.std(R), R


    # get reward from specific model
    def get_reward(self, traj, reward_idx):
        reward = []
        for idx in reward_idx:
            critic, _, _ = self.models[idx]
            reward.append(critic(traj).item() )
        return np.mean(reward), np.std(reward)
        critic, _ = self.models[idx]
        return critic(traj).item()


    # train all the reward models
    def update_parameters(self, memory, batch_size):
        loss = np.zeros((self.n_models,))
        for idx, (critic, optimizer, scheduler) in enumerate(self.models):
            loss[idx] = self.update_critic(critic, optimizer, scheduler, memory, batch_size)
        return np.mean(loss)


    # train a specific reward model
    def update_critic(self, critic, optimizer, scheduler, memory, batch_size):

        # sample a batch of (trajectory, reward) from memory
        trajs, rewards = memory.sample(batch_size)
        trajs = torch.FloatTensor(trajs).to(device=self.device)
        

        rewards = torch.FloatTensor(rewards).to(device=self.device).unsqueeze(1)
        
        # train critic using supervised approach
        rhat= critic(trajs).to(device=self.device)
        q_loss = F.mse_loss(rhat, rewards)
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        scheduler.step()
        return q_loss.item()

    def save_model(self, save_name:str):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, (critic, optimizer, schduler) in enumerate(self.models):
            torch.save(critic.state_dict(), save_dir + '/model_' + str(idx))