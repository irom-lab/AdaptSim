import numpy as np
import random


class UtilAdaptPendulum():
    """
    Utilities for adaptation in Pendulum.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.q_ind_all = np.linspace(0, cfg.max_step - 2, cfg.num_q, dtype=int)
        random.seed(cfg.seed)

    def scale_reward(self, reward, **kwargs):
        return reward / 2

    def get_target_obs(self, episodes_info, **kwargs):
        """
        Snapshots of joint angles.
        """

        assert len(episodes_info) == 1
        info = episodes_info[0]
        q_snapshot = np.vstack(info['q'])[self.q_ind_all].flatten()
        return q_snapshot[None]
