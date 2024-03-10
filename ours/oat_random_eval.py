import numpy as np
import torch
import gym
import robosuite as suite
from robosuite.controllers import load_controller_config
import torch.nn.functional as F
from scipy.optimize import minimize, LinearConstraint
import pickle
import argparse
import random
from tqdm import tqdm
from models import RNetwork
from scipy.spatial.transform import Rotation as R


class OAT(object):
    def __init__(self, state_dim, objs, num_wp, save_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        # self.objs = torch.FloatTensor(objs).to(device=self.device)
        self.wp_id = num_wp
        self.best_wp = []
        self.n_inits = 5
        self.lr = 1e-3
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.learned_models = []
        self.n_eval = 100

        save_dir = 'models/' + save_name

        for wp_id in range(1, self.wp_id+1):
            models = []
            for idx in range(self.n_models):
                critic = RNetwork(wp_id*self.state_dim + len(objs), hidden_dim=self.hidden_size).to(device=self.device)
                critic.load_state_dict(torch.load(save_dir + '/model_' + str(wp_id) + '_' + str(idx)))
                models.append(critic)
            self.learned_models.append(models)

        self.best_traj = 0.5*(np.random.rand(self.state_dim*self.wp_id) - 0.5)
        self.best_reward = -np.inf
        self.lincon = LinearConstraint(np.eye(self.state_dim), -0.5, 0.5)

    def traj_opt(self, objs):
        self.reward_idx = random.choice(range(self.n_models))
        self.objs = torch.FloatTensor(objs).to(device=self.device)

        self.traj = []
        for idx in range(1, self.wp_id+1):
            min_cost = np.inf
            self.curr_wp = idx-1

            for t_idx in range(self.n_inits):
                xi0 = np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim]) + np.random.normal(0, 0.1, size=self.state_dim) if t_idx!=0 else np.copy(self.best_traj[self.curr_wp*self.state_dim:self.curr_wp*self.state_dim+self.state_dim])

                res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon, options={'eps': 1e-6, 'maxiter': 1e6})
                if res.fun < min_cost:
                    min_cost = res.fun
                    self.best_wp = res.x

            self.traj.append(self.best_wp)
        return np.array(self.traj).flatten()

    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)

    def get_cost(self, traj):
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj_learnt = torch.FloatTensor(np.array(self.traj).flatten()).to(device=self.device)
        traj = torch.cat((traj_learnt, traj)).to(device=self.device)
        traj = torch.cat((traj, self.objs)).to(device=self.device)
        reward = self.get_reward(traj)
        return -reward
    
    def get_reward(self, traj):
        loss = 0
        models = self.learned_models[self.curr_wp]
        for idx in range (self.n_models):
            critic = models[idx]
            loss += critic(traj).item()
        return loss/self.n_models

    def get_avg_reward(self, traj):
        reward = 0
        traj = torch.FloatTensor(traj).to(device=self.device)
        traj = torch.cat((traj, self.objs))
        for idx in range(self.n_models):
            critic, _ = self.models[idx]
            reward += critic(traj).item()
        return reward/self.n_models

    def update_parameters(self, memory, batch_size):
        loss = np.zeros((self.n_models))
        for idx, (critic, optim) in enumerate(self.models):
            loss[idx] = self.update_critic(critic, optim, memory, batch_size)
        return np.mean(loss)



def run_ours(args):
    # training parameters
    save_data = {'episode': [], 'reward': [], 'best_traj': []}
    if args.object == 'test':
        save_name = args.env + '/' + args.run_num 
    else:
        save_name = args.env + '/' + args.object + '/' + args.run_num

    if args.use_latch and args.env=='Door':
        save_name = args.env + '/Door_with_latch' + '/' + args.run_num 
    elif not args.use_latch and args.env=='Door':
        save_name = args.env + '/Door_without_latch' + '/' + args.run_num 

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # create environment instance
    env = suite.make(
        env_name=args.env, 
        robots="Panda",  
        controller_configs=controller_config,
        has_renderer=args.render,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        single_object_mode=2,
        object_type=args.object,
        use_latch=args.use_latch,
    )

    # Agent
    num_wp = args.num_wp
    wp_id = num_wp
    obs = env.reset()
    if args.env == 'Stack':
        objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
    elif args.env == 'Lift':
        objs = obs['cube_pos']
    elif args.env == 'PickPlace':
        objs = obs[args.object+'_pos']
    elif args.env == 'NutAssembly':
        nut = 'RoundNut'
        objs = obs[nut + '_pos']
    elif args.env == 'Door':
        objs = np.concatenate((obs['door_pos'], obs['handle_pos']), axis=-1)

    if args.env != 'Door':    
        agent = OAT(state_dim=4, objs=objs, num_wp=num_wp, save_name=save_name)
    else:
        agent = OAT(state_dim=7, objs=objs, num_wp=num_wp, save_name=save_name)


    n_eval = args.n_eval

    # Main loop
    total_steps = 0
    for i_episode in tqdm(range(0, n_eval)):
       
        # initialize variables
        episode_reward = 0
        done, truncated = False, False
        obs = env.reset()

        if args.env == 'Stack':
            objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1) 
        elif args.env == 'Lift':
            objs = obs['cube_pos']
        elif args.env == 'PickPlace':
            objs = obs[args.object+'_pos']
        elif args.env == 'NutAssembly':
            nut = 'RoundNut'
            objs = obs[nut + '_pos']
        elif args.env == 'Door':
            objs = np.concatenate((obs['door_pos'], obs['handle_pos']), axis=-1)

        # select optimal trajectory
        traj_full = agent.traj_opt(objs)
        
        if args.env != 'Door':
            traj_mat = np.reshape(traj_full, (wp_id,4))[:, :3] + obs['robot0_eef_pos']
            gripper_mat = np.reshape(traj_full, (wp_id, 4))[:, 3:] 
        else: 
            robot_ee = obs['robot0_eef_pos']
            robot_quat = obs['robot0_eef_quat']

            r = R.from_quat(robot_quat)
            robot_eul = r.as_euler('xyz', degrees=False)
            
            traj_mat = np.reshape(traj_full, (wp_id,7))[:, :6] + np.concatenate((robot_ee, robot_eul), axis=-1)
            gripper_mat = np.reshape(traj_full, (wp_id, 7))[:, 6:] 
            
        
        time_s = 0
        train_reward = 0
        for timestep in range(wp_id*50):

            if args.render:
                env.render()    # toggle this when we don't want to render

            # convert traj to actions
            if args.env != 'Door':
                state = obs['robot0_eef_pos']
            else:
                robot_ee = obs['robot0_eef_pos']
                robot_quat = obs['robot0_eef_quat']

                r = R.from_quat(robot_quat)
                robot_eul = r.as_euler('xyz', degrees=False)
                state = np.concatenate((obs['robot0_eef_pos'], robot_eul), axis=-1)

            widx = int(np.floor(timestep / (50)))
            error = traj_mat[widx, :] - state

            if args.env != 'Door':
                if timestep < 10:
                    full_action = np.array(list(10. * error) + [0.]*3 + [-1.])
                elif time_s >= 35:
                    full_action = np.array([0.]*6 + list(gripper_mat[widx]))
                else:
                    full_action = np.array(list(10. * error) + [0.]*4)
            else:
                if timestep < 10:
                    full_action = np.array(list(10. * error) + [-1.])
                elif time_s >= 35:
                    # full_action = np.array(list(10. * error) + [0.]*3 + list(gripper_mat[widx]))
                    full_action = np.array([0.]*6 + list(gripper_mat[widx]))
                else:
                    full_action = np.array(list(10. * error) + [0.])

            time_s += 1
            if time_s >= 50:
                time_s = 1

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward
            if timestep//50 == wp_id-1:
                train_reward += reward
            total_steps += 1
        
        save_data['episode'].append(i_episode)
        save_data['reward'].append(episode_reward)

    pickle.dump(save_data, open('models/' + save_name + '/eval_reward.pkl', 'wb'))
    print("Reward data saved in {}".format(save_name) + '/eval_reward.pkl')



if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True)
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')
    p.add_argument('--n_inits', type=int, default=1)
    p.add_argument('--num_wp', type=int, default=5)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--render', action='store_true', default=False)
    p.add_argument('--use_latch', action='store_true', default=False)
    args = p.parse_args()
    run_ours(args)

