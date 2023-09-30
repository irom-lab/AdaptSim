import numpy as np
import random


class UtilAdaptDP():
    """
    Utilities for adaptation in Double Pendulum.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.target_obs = cfg.target_obs
        self.reward_scaling = cfg.reward_scaling
        self.qdot_scaling = np.array([cfg.q0dot_scaling, cfg.q1dot_scaling])
        self.q_ind_all = np.array(
            [-1 - ind * cfg.skip for ind in range(0, cfg.traj_len)], dtype=int
        )  # from last
        random.seed(cfg.seed)

    def scale_reward(self, reward, **kwargs):
        return reward * self.reward_scaling

    def get_target_obs(self, episodes_info, **kwargs):
        """
        Snapshots of joint angles.
        """

        assert len(episodes_info) == 1
        info = episodes_info[0]

        if self.target_obs == 'q_snapshot':
            out = np.vstack(info['q'])[self.q_ind_all].flatten()
        elif self.target_obs == 'q_and_a_snapshot':
            q = np.vstack(info['q'])[self.q_ind_all].flatten()
            a = np.vstack(info['a'])[self.q_ind_all] * self.qdot_scaling
            a = a.flatten()
            out = np.hstack((q, a))
        else:
            raise "Unknown target obs type for DP!"
        return out[None]
