import numpy as np
import torch
from memory import MyMemory
from method import Method
import gym
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_controller_config
import pickle
import argparse

def run_ours(args):
    # training parameters
    save_data = {'episode': [], 'reward': [], 'best_traj': []}
    if args.object == 'test':
        save_name = 'Stack/' + args.run_num 
    batch_size = 30

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # create environment instance
    env = suite.make(
        env_name="Stack", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=controller_config,
        has_renderer=False,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        # use_latch=False,
    )

    # Agent
    num_wp = args.num_wp
    n_inits = args.n_inits
    obs = env.reset()
    objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']))
    agent = Method(traj_dim=4*num_wp, state_dim=4, n_traji=n_inits, objs=objs)

    # Memory
    memory = MyMemory()

    # Logger
    run_name = run_name = 'runs/ours_' + args.run_num + datetime.datetime.now().strftime("%H-%M")
    writer = SummaryWriter(run_name)


    # Main loop
    total_steps = 0
    timestep = 1
    mean_episode_reward = 0
    std_episode_reward = 0
    for i_episode in range(1, 2501):

        # initialize variables
        episode_reward = 0
        done, truncated = False, False
        obs = env.reset()

        # select optimal trajectory
        # traj = 0.5*(np.random.rand(15)-0.5)
        traj = 0.5*(np.random.rand(4*num_wp)-0.5)
        if i_episode > 40:
            traj = agent.traj_opt(i_episode)
        traj_mat = np.reshape(traj, (num_wp,4))[:, :3] + obs['robot0_eef_pos']
        gripper_mat = np.reshape(traj, (num_wp, 4))[:, 3:]

        if i_episode > 100 and i_episode < 400 and std_episode_reward < 5:
            wp_id = np.random.randint(num_wp)
            traj_mat[wp_id] += np.random.normal(0, 0.1, size=len(traj_mat[wp_id]))
            gripper_mat[wp_id] += np.random.normal(0, 0.1)

        if i_episode > 400 and i_episode < 900 and std_episode_reward < 2:
            wp_id = num_wp-1 # np.random.randint(num_wp)
            traj_mat[wp_id] += np.random.normal(0, 0.1, size=len(traj_mat[wp_id]))
            gripper_mat[wp_id] += np.random.normal(0, 0.1)
    
        for timestep in range(100):

            # env.render()    # toggle this when we don't want to render

            if len(memory) > batch_size:
                for _ in range(1):
                    critic_loss = agent.update_parameters(memory, batch_size)
                    writer.add_scalar('model/critic', critic_loss, total_steps)



            # convert traj to actions
            state = obs['robot0_eef_pos']
            g_state = obs['robot0_gripper_qpos']
            widx = int(np.floor(timestep / (100/num_wp)))
            error = traj_mat[widx, :] - state
            if timestep <= 8:
                full_action = np.array([0.]*6 + list(gripper_mat[widx]))
            else:
                full_action = np.array(list(10. * error) + [0.]*3 + list(gripper_mat[widx]))
            
            timestep += 1
            if timestep >= (100/num_wp):
                timestep = 1

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward
            total_steps += 1

        memory.push(np.concatenate((traj, objs)), episode_reward)
        save_data['episode'].append(i_episode)
        save_data['reward'].append(episode_reward)

        if episode_reward > agent.best_reward:
            agent.set_init(traj, episode_reward)
            print(episode_reward, "new best trajectory")
            agent.save_model(save_name)
            save_data['best_traj'].append(traj)
            pickle.dump(traj, open('models/' + save_name + '/traj.pkl', 'wb'))

        if i_episode > 10:
            mean_episode_reward = np.mean(save_data['reward'][-10:])
            std_episode_reward = np.std(save_data['reward'][-10:])
        
        # if i_episode%100 == 0:
        #     print("[*] Evaluating and updating best model")
        #     agent.eval_best_model(memory, save_name)
        
        writer.add_scalar('reward', episode_reward, i_episode)
        print("Episode: {}, Reward: {}".format(i_episode, round(episode_reward, 2)))

    pickle.dump(save_data, open('models/' + save_name + '/data.pkl', 'wb'))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')
    p.add_argument('--n_inits', type=int, default=1)
    p.add_argument('--num_wp', type=int, default=5)
    args = p.parse_args()
    run_ours(args)


"""
Why not formulate our approach as a Hirarchical Learning approach where we learn the subtasks using our approach (which we have seen works) and then use a higher level policy to convert these subtasks to a main task?

This way we can leverage the high speed learning from our proposed approach and to train the low level policies and reduce the learning time of the Hirarchical methods that use those policies.
"""