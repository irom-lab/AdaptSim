"""
Utilities for adaptation in Linearized Double Pendulum task.

"""
import numpy as np
import random


class UtilAdaptDPLinearized():

    def __init__(self, cfg):
        self.cfg = cfg
        self.target_obs = cfg.target_obs
        self.reward_scaling = cfg.reward_scaling
        self.a_scaling = np.array(cfg.a_scaling)
        self.step_ind_all = np.array(
            [ind * (cfg.skip + 1) for ind in range(1, cfg.traj_len + 1)],
            dtype=int
        )  # from beginning, do not use first step
        random.seed(cfg.seed)

    def scale_reward(self, reward, pre_reward=None):
        # assume reward negative!
        if reward > 0 or pre_reward > 0:
            raise 'DP linearized reward should be negative!'
        return 1 - self.reward_scaling * (reward-pre_reward) / pre_reward

    def get_target_obs(self, episodes_info, *kwargs):
        """
        Snapshots of joint angles.
        """
        assert len(episodes_info) == 1
        info = episodes_info[0]

        if self.target_obs == 'q_snapshot':
            q = np.vstack(info['x'])[self.step_ind_all, :2]
            q[:, 0] = (q[:, 0] + 3.14) / 3.14
            out = q.flatten()
        elif self.target_obs == 'q_and_a_snapshot':
            q = np.vstack(info['x'])[self.step_ind_all, :2]
            q[:, 0] = (q[:, 0] + 3.14) / 3.14
            q = q.flatten()
            a = np.vstack(info['u'])[self.step_ind_all] * self.a_scaling
            a = a.flatten()
            out = np.hstack((q, a))
        elif self.target_obs == 'q_full_snapshot':
            q = np.vstack(info['x'])[self.step_ind_all]
            q[:, 0] = (q[:, 0] + 3.14) / 3.14
            q[:, 2] /= 5  # normalize qdot1
            out = q.flatten()
        elif self.target_obs == 'q_full_and_a_snapshot':
            q = np.vstack(info['x'])[self.step_ind_all]
            q[:, 0] = (q[:, 0] + 3.14) / 3.14
            q[:, 2] /= 5  # normalize qdot1
            q = q.flatten()
            a = np.vstack(info['u'])[self.step_ind_all] * self.a_scaling
            a = a.flatten()
            out = np.hstack((q, a))
        elif self.target_obs == 'q_and_a_snapshot_and_reward':
            q = np.vstack(info['x'])[self.step_ind_all, :2]
            q[:, 0] = (q[:, 0] + 3.14) / 3.14
            q = q.flatten()
            a = np.vstack(info['u'])[self.step_ind_all] * self.a_scaling
            a = a.flatten()
            out = np.hstack((q, a, self.scale_reward(info['reward'])))
            raise 'Need to scale reward!'
        else:
            raise "Unknown target obs type for DP!"
        return out[None]
