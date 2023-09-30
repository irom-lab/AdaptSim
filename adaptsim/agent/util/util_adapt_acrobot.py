"""
Utilities for adaptation in Acrobot task.

"""
import numpy as np
import random


class UtilAdaptAcrobot():

    def __init__(self, cfg):
        self.cfg = cfg
        self.q_ind_all = np.array([
            ind * cfg.skip - 1 for ind in range(1, cfg.traj_len + 1)
        ], dtype=int)
        random.seed(cfg.seed)

    def scale_reward(self, reward, pre_reward=None):
        return reward

    def get_target_obs(self, episodes_info, **kwargs):
        """
        Snapshots of joint angles.
        """

        assert len(episodes_info) == 1
        info = episodes_info[0]
        x_snapshot = np.vstack(info['x'])[self.q_ind_all, :2].flatten()
        return x_snapshot[None]
