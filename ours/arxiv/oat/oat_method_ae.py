import numpy as np
import pickle
import torch
import torch.nn.functional as F
import os, sys
import random
from models import AE


class OAT(object):
    def __init__(self, state_dim, objs, wp_id, save_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.objs = torch.FloatTensor(objs).to(device=self.device)
        self.wp_id = wp_id
        self.best_wp = []
        self.n_inits = 5
        self.lr = 1e-3
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.learned_models = []
        self.n_eval = 100

        for _ in range(self.n_models):
            critic = AE((self.wp_id-1)*self.state_dim + len(objs), hidden_dim=self.hidden_size, out_dim=4).to(device=self.device)
            optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            self.models.append((critic, optimizer))


        save_dir = 'models/' + save_name

        # for w_id in range(1, self.wp_id):
        #     models = []
        #     for idx in range(self.n_models):
        #         critic = AE((w_id-1)*self.state_dim + len(objs), hidden_dim=self.hidden_size, out_dim=4).to(device=self.device)
        #         critic.load_state_dict(torch.load(save_dir + '/model_' + str(w_id) + '_' + str(idx)))
        #         models.append(critic)
        #     self.learned_models.append(models)

        for w_id in range(1, self.wp_id):
            models = []
            critic = AE((w_id-1)*self.state_dim + len(objs), hidden_dim=self.hidden_size, out_dim=4).to(device=self.device)
            critic.load_state_dict(torch.load(save_dir + '/model_best' + str(w_id)))
            self.learned_models.append(critic)


        self.best_traj = 0.5*(np.random.rand(self.state_dim*self.wp_id) - 0.5)
        self.best_reward = -np.inf

    def traj_opt(self, i_episode, objs):
        self.objs = torch.FloatTensor(objs).to(device=self.device)
        min_reward = -np.inf
        self.traj = torch.FloatTensor([]).to(device=self.device)
        best_wp = None
        for idx in range(1, self.wp_id+1):
            self.reward_idx = random.choice(range(self.n_models))

            self.load_model = True if idx!=self.wp_id else False
            self.curr_wp = idx-1

            if idx == self.wp_id and i_episode <= 75:
                wp = torch.FloatTensor(0.5*(np.random.rand(4) - 0.5)).to(device=self.device)
            else:
                wp, _ = self.get_wp()

            # if idx == self.wp_id and np.random.rand() < 25/(1+i_episode):

            #     wp += torch.randn_like(wp)*0.05
            critic, _ = self.models[self.reward_idx]
            if idx == self.wp_id:
                for t_idx in range(self.n_inits):
                    wp0 = torch.clone(wp) + torch.randn_like(wp)*0.05 if t_idx!=0 else torch.clone(wp)

                    reward = critic.decoder(wp0)
                    if reward > min_reward:
                        min_reward = reward
                        best_wp = torch.clone(wp)
            else:
                best_wp = torch.clone(wp)
                min_reward = critic.decoder(best_wp)

            
            self.traj = torch.cat((self.traj, best_wp)).to(device=self.device)
        return self.traj, min_reward

    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)

    def get_wp(self):
        if self.load_model:
            # models = self.learned_models[self.curr_wp]
            # wp_out = torch.zeros(4).to(device=self.device)
            # reward_out = 0
            # for idx in range (self.n_models):
            #     critic = models[idx]
            #     input = self.traj[:self.state_dim*self.curr_wp]
            #     input = torch.cat((self.objs, input)).to(device=self.device)
            #     wp, reward = critic(input)
            #     wp_out += wp
            #     reward_out += reward
            # return wp_out/self.n_models, reward_out/self.n_models
            critic = self.learned_models[self.curr_wp]
            input = self.traj[:self.state_dim*self.curr_wp]
            input = torch.cat((self.objs, input)).to(device=self.device)
            wp, reward = critic(input)
            return wp, reward


        else: 
            critic, _ = self.models[self.reward_idx]
            input = self.traj[:self.state_dim*self.curr_wp]
            input = torch.cat((self.objs, input)).to(device=self.device)
            wp, reward = critic(input)
            return wp, reward

    
    def update_parameters(self, memory, batch_size, i_episode):
        loss = np.zeros((self.n_models))
        for idx, (critic, optim) in enumerate(self.models):
            loss[idx] = self.update_critic(critic, optim, memory, batch_size, i_episode)
        return np.mean(loss)

    def update_critic(self, critic, optim,  memory, batch_size, i_episode):
        start, wp, rewards_true = memory.sample(batch_size)
        start = torch.FloatTensor(start).to(device=self.device)
        rewards_true = torch.FloatTensor(rewards_true).to(device=self.device)
        wp = torch.FloatTensor(wp).to(device=self.device)
        _, rewards1 = critic(start)
        rewards2 = critic.decoder(wp)
        # rewards_ae = critic.decoder(trajs).to(device=self.device)
        l1 = -torch.tanh(torch.mean(rewards1))
        l2 = F.mse_loss(rewards2, rewards_true)
        loss = 2*l1 + l2 if i_episode>=125 else l2
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss.item()


    def eval_best_model(self, memory, save_name:str):
        loss_arr = np.zeros(self.n_models)
        start, wp, rewards_true = memory.sample(self.n_eval)
        start = torch.FloatTensor(start).to(device=self.device)
        rewards_true = torch.FloatTensor(rewards_true).to(device=self.device)
        wp = torch.FloatTensor(wp).to(device=self.device)

        for idx, (critic, optimizer) in enumerate(self.models):
            _, rewards1 = critic(start)
            rewards2 = critic.decoder(wp)
            l1 = -torch.tanh(torch.mean(rewards1))
            l2 = F.mse_loss(rewards2, rewards_true)
            loss = l1 + l2
            loss_arr[idx] = loss.cpu().detach().numpy()
        
        self.best_model = np.argmin(loss_arr)
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        critic, _ = self.models[self.best_model]
        torch.save(critic.state_dict(), save_dir + '/model_best' + str(self.wp_id))

        self.best_model = np.argmin(loss_arr)

        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        critic, _ = self.models[self.best_model]
        torch.save(critic.state_dict(), save_dir + '/model_best' + str(self.wp_id))


    def reset_model(self, idx):
        critic = AE(len(self.objs), hidden_dim=self.hidden_size, out_dim=self.state_dim).to(device=self.device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr) 
        self.models[idx] = (critic, optimizer)
        print("RESET MODEL ", idx)


    def save_model(self, save_name:str):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, (critic, optimizer) in enumerate(self.models):
            torch.save(critic.state_dict(), save_dir + '/model_' + str(self.wp_id) + '_' + str(idx))





        