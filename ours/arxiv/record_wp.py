import numpy as np
import pickle
import gym 
import argparse
import pygame
import time
import os, sys

import robosuite as suite
from robosuite.controllers import load_controller_config


class Joystick(object):
    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1
        self.timeband = 0.5
        self.lastpress = time.time

    def input(self):
        pygame.event.get()
        curr_time = time.time()
        z1 = self.gamepad.get_axis(1)
        z2 = self.gamepad.get_axis(0)
        z3 = -self.gamepad.get_axis(4)
        if abs(z1)<self.deadband:
            z1 = 0.0
        if abs(z2)<self.deadband:
            z2 = 0.0
        if abs(z3)<self.deadband:
            z3 = 0.0

        A_pressed = self.gamepad.get_button(0) #and (curr_time - self.lastpress > self.timeband)
        B_pressed = self.gamepad.get_button(1) #and (curr_time - self.lastpress > self.timeband)
        X_pressed = self.gamepad.get_button(2) #and (curr_time - self.lastpress > self.timeband)
        Y_pressed = self.gamepad.get_button(3) #and (curr_time - self.lastpress > self.timeband)
        START_pressed = self.gamepad.get_button(7) #and (curr_time - self.lastpress  >self.timeband)
        Right_trigger = self.gamepad.get_button(5)
        Left_trigger = self.gamepad.get_button(4)
        if A_pressed or B_pressed or START_pressed or X_pressed or Y_pressed:
            self.lastpress = curr_time
        return [z1, z2, z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, Right_trigger, Left_trigger


class record_wp():
    def __init__(self, args):
        self.save_data = {'wp': [], 'reward': []}
        if args.object == 'test':
            self.save_name = args.env + '/' + args.run_num + '_demo.pkl'
        else:
            self.save_name = args.env + '/' + args.object + '/' + args.run_num + '_demo.pkl'

        if not os.path.exists('demos/' + args.env):
            os.makedirs('demos/' + args.env)

        self.controller_config = load_controller_config(default_controller = "OSC_POSE")
        
        # create environment instance
        self.env = suite.make(
        env_name=args.env, # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        controller_configs=self.controller_config,
        has_renderer=True,
        reward_shaping=False,
        control_freq=10,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        single_object_mode=2,
        object_type=args.object,
        use_latch=False,
        horizon=1000000000
        )

        self.teleop(args)

    def teleop(self, args):
        interface = Joystick()
        action_scale = 0.1
        obs = self.env.reset()
        if args.env == 'Lift':
            obj_pos = obs['cube_pos']
        elif args.env == 'Stack':
            obj_pos = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1)
        elif args.env == 'PickPlace':
            obj_pos = obs[args.object + '_pos']

        gripper_action = -1

        print("[*] Ready for Teleoperation...")

        time_s = 0
        total_reward = 0
        while True:
            timestamp = time.time()
            state = obs['robot0_eef_pos']

            g_state = obs['robot0_gripper_qpos']

            z, A_pressed, _, X_pressed, _, START_pressed, _, _ = interface.input()

            if START_pressed:
                print('[*] DONE !!!')
                self.save_data['wp'] = np.concatenate((self.save_data['wp'], obj_pos))
                print(self.save_data)
                pickle.dump(self.save_data, open('demos/' + self.save_name,'wb'))
                return False

            if A_pressed:
                wp = np.concatenate((state, [gripper_action]))
                self.save_data['wp'] = np.concatenate((self.save_data['wp'], wp))
                self.save_data['reward'].append(total_reward/time_s*50)
                time_s = 0
                total_reward = 0
                time.sleep(0.5)
            
            if X_pressed:
                gripper_action *= -1
                time.sleep(0.5)

            full_action = np.array(list(action_scale*np.array(z)) + [0]*3 + [gripper_action])
            obs, reward, done, _ = self.env.step(full_action)

            total_reward += reward
            time_s += 1

            self.env.render()

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, required=True)
    p.add_argument('--run_num', type=str, default='test')
    p.add_argument('--object', type=str, default='test')

    args = p.parse_args()
    record_wp(args)