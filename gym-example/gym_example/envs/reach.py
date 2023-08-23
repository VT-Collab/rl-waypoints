import numpy as np
import gym
from gym import spaces


class ReachDense(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(
            low=-0.2,
            high=+0.2,
            shape=(2,),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(2,),
            dtype=np.float64
        )
        self.goal1 = np.array([0.8, +0.5])
        self.goal2 = np.array([0.8, -0.5])
        self.switch = False

    def _get_obs(self):
        return np.copy(self.robot)

    def reset(self, seed=None, options=None):
        self.robot = np.array([0., 0.])
        self.switch = False
        return self._get_obs(), {}

    def step(self, action):
        self.robot += action
        if np.linalg.norm(self.goal1 - self.robot) < 0.2:
            self.switch = True
        if not self.switch:
            reward = -np.linalg.norm(self.goal1 - self.robot) * 100
        else:
            reward = -np.linalg.norm(self.goal2 - self.robot) * 200 + 100
        return self._get_obs(), reward, False, False, {}