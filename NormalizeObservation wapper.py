import gym
import numpy as np
import air_conditioning_env as ac_env

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)
        # 獲取原始觀察空間的最小值和最大值
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high

    def observation(self, observation):
        # 使用 min-max normalization 將觀察值歸一化到 [0, 1] 範圍
        normalized_obs = (observation - self.obs_low) / (self.obs_high - self.obs_low)
        return normalized_obs
