import numpy as np
import logging
import random

from util.numeric import normalize


class UtilAdaptScoop():
    """
    Utilities for scooping with fixed initialization.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg.seed)
        self.obs_lb = np.array(cfg.obs_lb)
        self.obs_ub = np.array(cfg.obs_ub)
        self.num_episode_for_obs = cfg.num_episode_for_obs

    def scale_reward(self, reward, **kwargs):
        if self.cfg.reward_scaling == 'raw':
            return reward
        else:
            logging.error("Unknown reward scaling!")
            raise

    def get_target_obs(self, episodes_info):
        random.shuffle(episodes_info)
        if len(episodes_info) > self.num_episode_for_obs:
            episodes_info = random.sample(
                episodes_info, self.num_episode_for_obs
            )

        if self.cfg.target_obs == 'traj':
            obs_all = []
            for episode_info in episodes_info:
                obs = np.vstack(episode_info['veggie_xy_all'])
                obs = normalize(obs, self.obs_lb, self.obs_ub)
                obs_all += [obs.flatten()]
            obs_all = np.concatenate(obs_all)
            return obs_all[None]
        elif self.cfg.target_obs == 'traj_plus_reward':
            obs_all = []
            for episode_info in episodes_info:
                obs = np.vstack(episode_info['veggie_xy_all'])
                obs = normalize(obs, self.obs_lb, self.obs_ub)
                obs_all += [np.append(obs.flatten(), episode_info['reward'])]
            obs_all = np.concatenate(obs_all)
            return obs_all[None]
        else:
            logging.error("Unknown target obs type!")
            raise
