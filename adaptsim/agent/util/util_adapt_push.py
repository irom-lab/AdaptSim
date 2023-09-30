import numpy as np
import logging
import random

from util.numeric import normalize


class UtilAdaptPush():
    """
    Utilities for pushing with fixed initialization.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg.seed)
        self.num_episode_for_obs = cfg.num_episode_for_obs

        # Initialize observation normalization factors if needed
        if cfg.target_obs in [
            'final_T',
            'final_T_plus_goal',
            'final_T_plus_reward',
            'final_offset',
        ]:
            self.obs_lb = np.array(cfg.obs_lb)
            self.obs_ub = np.array(cfg.obs_ub)

    def scale_reward(self, reward, **kwargs):
        """
        Right now reward from env is scaled to [0, 1]
        """
        if self.cfg.reward_scaling == 'pre_reward':
            logging.error("Do not scale reward for now!")
            raise
        elif self.cfg.reward_scaling == 'raw':
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

        if self.cfg.target_obs == 'traj_plus_goal':
            # always keep the last step
            obs_all = []
            for episode_info in episodes_info:
                obs = np.vstack(episode_info['bottle_T_all'])[:, :2]
                if len(
                    obs
                ) > self.cfg.min_step:  # in case sim error and not enough steps stored
                    obs = obs[self.cfg.min_step:]
                steps = np.linspace(
                    0,
                    len(obs) - 1, num=self.cfg.num_step, endpoint=True,
                    dtype=np.int
                )  # end is inclusive
                obs = np.vstack((obs[steps], episode_info['goal']))
                obs = obs.flatten()

                # obs = normalize(obs, self.obs_lb, self.obs_ub)
                obs_all += [obs]
            obs_all = np.concatenate(obs_all)
            return obs_all[None]

        elif self.cfg.target_obs == 'final_T':
            obs_all = []
            for episode_info in episodes_info:
                obs = np.array(episode_info['bottle_T_final'][:2])
                obs = normalize(obs, self.obs_lb, self.obs_ub)
                obs_all += [obs]
            obs_all = np.concatenate(obs_all)
            return obs_all[None]

        elif self.cfg.target_obs == 'final_T_plus_goal':
            obs_all = []
            for episode_info in episodes_info:
                obs = np.hstack(
                    (episode_info['bottle_T_final'][:2], episode_info['goal'])
                )  # 4-dim
                obs = normalize(obs, self.obs_lb, self.obs_ub)
                obs_all += [obs]
            obs_all = np.concatenate(obs_all)
            return obs_all[None]

        elif self.cfg.target_obs == 'final_T_plus_reward':
            obs_all = []
            for episode_info in episodes_info:
                obs = np.append(
                    episode_info['bottle_T_final'][:2], episode_info['reward']
                )
                obs = normalize(obs, self.obs_lb, self.obs_ub)
                obs_all += [obs]
            obs_all = np.concatenate(obs_all)
            return obs_all[None]

        elif self.cfg.target_obs == 'final_offset':
            obs_all = []
            for episode_info in episodes_info:
                obs = episode_info['goal'] - episode_info['bottle_T_final'][:2]
                obs = normalize(obs, self.obs_lb, self.obs_ub)
                obs_all += [obs]
            obs_all = np.concatenate(obs_all)
            return obs_all[None]

        else:
            logging.error("Unknown target obs type!")
            raise
