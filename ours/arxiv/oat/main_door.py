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
from scipy.spatial.transform import Rotation as R
import argparse

def run_ours(args):
    # training parameters
    save_data = {'episode': [], 'reward': [], 'best_traj': []}
    if args.object == 'test':
        save_name = 'Door/' + args.run_num 
    batch_size = 30

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # create environment instance
    env = suite.make(
        env_name="Door", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=controller_config,
        has_renderer=False,
        reward_shaping=True,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        use_latch=True,
    )

    # Agentnum_wp = args.num_wp
    num_wp = args.num_wp
    n_inits = args.n_inits
    agent = Method(traj_dim=6*num_wp, state_dim=6, n_traji=n_inits)

    # Memory
    memory = MyMemory()

    # Logger
    run_name = 'runs/ours_' + datetime.datetime.now().strftime("%H-%M")
    writer = SummaryWriter(run_name)


    # Main loop
    total_steps = 0
    for i_episode in range(1, 10001):

        # initialize variables
        episode_reward = 0
        done, truncated = False, False
        obs = env.reset()

        lin = obs['robot0_eef_pos']
        quat = obs['robot0_eef_quat']
        r = R.from_quat(quat)
        ang_eul = r.as_euler('xyz', degrees=False)
        pos = np.concatenate((lin, ang_eul))

        # select optimal trajectory
        traj = 0.5*(np.random.rand(6*num_wp)-0.5)
        if i_episode > 40:
            traj = agent.traj_opt()
        traj_mat = np.reshape(traj, (num_wp,6)) + pos
    
        for timestep in range(100):

            # env.render()    # toggle this when we don't want to render

            if len(memory) > batch_size:
                for _ in range(1):
                    critic_loss = agent.update_parameters(memory, batch_size)
                    writer.add_scalar('model/critic', critic_loss, total_steps)

            # convert traj to actions
            state = pos
            widx = int(np.floor(timestep / (100/num_wp)))
            error = traj_mat[widx, :] - state
            full_action = np.array(list(10. * error) + [-1.])

            # take step
            obs, reward, done, _ = env.step(full_action)
            episode_reward += reward
            total_steps += 1

        memory.push(traj, episode_reward)
        save_data['episode'].append(i_episode)
        save_data['reward'].append(episode_reward)

        if episode_reward > agent.best_reward:
            agent.set_init(traj, episode_reward)
            print(episode_reward, "new best trajectory")
            agent.save_model(save_name)
            save_data['best_traj'].append(episode_reward)

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